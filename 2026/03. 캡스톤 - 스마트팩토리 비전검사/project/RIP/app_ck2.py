import os, time, threading, psutil, cv2, math, serial, logging
import numpy as np
import pyrealsense2 as rs
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from datetime import datetime
from queue import Queue, Empty
from collections import deque, Counter
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import google.genai as genai

# ✅ [1] 시스템 로깅 (품질 및 에러 추적)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("IndustrialMaster")

# ✅ [2] TensorRT 10 고성능 엔진 (스레드 세이프)
class TensorRTInferenceEngine:
    def __init__(self, engine_path):
        self.trt_logger = trt.Logger(trt.Logger.ERROR)
        self.cfx = cuda.Device(0).make_context()
        try:
            with open(engine_path, "rb") as f, trt.Runtime(self.trt_logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()
            self.allocate_buffers()
            logger.info("✅ TensorRT AI Engine Loaded.")
        except Exception as e:
            logger.error(f"❌ AI Engine Error: {e}")
            raise
        finally: self.cfx.pop()

    def allocate_buffers(self):
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = self.engine.get_tensor_dtype(name)
            host_mem = cuda.pagelocked_empty(trt.volume(shape), trt.nptype(dtype))
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append({'host': host_mem, 'device': device_mem})
                self.input_w, self.input_h = shape[3], shape[2]
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def infer(self, frame):
        self.cfx.push()
        try:
            blob = cv2.resize(frame, (self.input_w, self.input_h))
            blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            blob = blob.transpose(2, 0, 1).copy()[np.newaxis, :]
            self.inputs[0]['host'] = np.ascontiguousarray(blob)
            cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
            for i in range(self.engine.num_io_tensors):
                self.context.set_tensor_address(self.engine.get_tensor_name(i), self.bindings[i])
            self.context.execute_async_v3(self.stream.handle)
            for out in self.outputs:
                cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
            self.stream.synchronize()
            return [out['host'].copy() for out in self.outputs]
        finally: self.cfx.pop()

# ✅ [3] 스마트 팩토리 통합 관리자
class SmartFactoryManager:
    def __init__(self):
        self.stats = {"cpu": 0, "memory": 0, "fps": 0, "good": 0, "defect": 0}
        self.latest_results = []
        self.locked_target = None
        self.detection_log = deque(maxlen=20)
        self.vote_buffer = deque(maxlen=7)
        # 큐 크기를 1로 제한하여 '최신 프레임' 우선 전략 (카메라 안 보임 해결)
        self.frame_queue = Queue(maxsize=1)
        self.display_queue = Queue(maxsize=1)
        self.ser = None
        self.intrinsics = None
        self.lock = threading.Lock()
        self.is_running = True

    def init_arduino(self, port='/dev/ttyACM0'):
        try:
            self.ser = serial.Serial(port, 9600, timeout=1)
            logger.info(f"✅ Arduino Connected: {port}")
        except: logger.warning("⚠️ Arduino Serial Not Found.")

mgr = SmartFactoryManager()

# ✅ [4] 후처리 및 정밀 XYZ 계산 (번짐 방지 포함)
def postprocess_industrial(outputs, w, h, depth_f, intrinsics):
    preds, proto = outputs[0], outputs[1]
    num_ch = preds.shape[0] // 8400
    preds = preds.reshape(num_ch, -1).T
    proto = proto.reshape(32, 160 * 160)
    
    indices = cv2.dnn.NMSBoxes(preds[:, :4].tolist(), preds[:, 4:num_ch-32].max(axis=1).tolist(), 0.5, 0.4)
    res = []
    for i in indices:
        idx = int(i)
        xc, yc, bw, bh = preds[idx, :4]
        conf = float(preds[idx, 4:num_ch-32].max())
        x1, y1 = int((xc - bw/2) * (w/640)), int((yc - bh/2) * (h/640))
        bw_s, bh_s = int(bw * (w/640)), int(bh * (h/640))

        # 마스크 번짐 방지 최적화
        m_raw = np.matmul(preds[idx, num_ch-32:].reshape(1, 32), proto).reshape(160, 160)
        m_bin = (cv2.resize(1/(1+np.exp(-m_raw)), (w, h)) > 0.5).astype(np.uint8)
        crop = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(crop, (max(0, x1), max(0, y1)), (min(w, x1+bw_s), min(h, y1+bh_s)), 1, -1)
        m_bin &= crop
        
        contours, _ = cv2.findContours(m_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            area = cv2.contourArea(contours[0])
            rect = area / (bw_s * bh_s + 1e-6)
            cx, cy = x1 + bw_s//2, y1 + bh_s//2
            
            # 정밀 XYZ 보정 로직
            try:
                dist = depth_f.get_distance(cx, cy)
                if dist == 0: dist = depth_f.get_distance(cx+2, cy+2)
                pt = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], dist)
                xyz = (round(pt[0]*100, 1), round(pt[1]*100, 1), round(pt[2]*100, 1))
            except: xyz = (0.0, 0.0, 0.0)
            
            size = "S" if area < 52100 else "B" if area > 260000 else "M"
            state = "Normal" if rect > 0.5 else "Abnormal"
            res.append({"box": [x1, y1, bw_s, bh_s], "conf": conf, "class": f"{size}_{state}",
                        "contours": contours, "center": (cx, cy), "coords": xyz})
    return res

# ✅ [5] 고성능 워커 스레드 (병목 방지 로직)
def camera_worker():
    logger.info("📸 Camera Worker Thread Started.")
    pipeline = rs.pipeline(); config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    try:
        profile = pipeline.start(config)
        mgr.intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        align = rs.align(rs.stream.color)
        while mgr.is_running:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            aligned = align.process(frames)
            color_img = np.asanyarray(aligned.get_color_frame().get_data())
            depth_data = aligned.get_depth_frame()
            
            # 큐가 꽉 찼으면 가장 오래된 걸 버리고 최신을 넣음 (화면 밀림 방지)
            if mgr.frame_queue.full():
                try: mgr.frame_queue.get_nowait()
                except: pass
            mgr.frame_queue.put((color_img, depth_data))
    except Exception as e: logger.error(f"Camera Crash: {e}")

def inference_worker(ai):
    logger.info("🧠 Inference Worker Thread Started.")
    fps_time = time.time()
    while mgr.is_running:
        try:
            img, depth = mgr.frame_queue.get(timeout=2)
            out = ai.infer(img)
            objs = postprocess_industrial(out, img.shape[1], img.shape[0], depth, mgr.intrinsics)
            
            # 🎨 시각화 렌더링 (X, Y, Z 및 정확도 강제 출력)
            for item in objs:
                x, y, bw, bh = item["box"]
                cv2.drawContours(img, item["contours"], -1, (0, 255, 0), 2)
                overlay = img.copy()
                cv2.fillPoly(overlay, item["contours"], (0, 255, 0))
                cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)
                
                # 라벨 및 좌표 텍스트
                xyz = item["coords"]
                label = f"{item['class']} ({item['conf']:.2f})"
                coord_text = f"X:{xyz[0]} Y:{xyz[1]} Z:{xyz[2]}cm"
                
                ty = y - 35 if y > 50 else y + 20
                cv2.putText(img, label, (x, ty), 0, 0.6, (0, 255, 0), 2)
                cv2.putText(img, coord_text, (x, ty + 20), 0, 0.6, (255, 255, 0), 2) # Z 포함 좌표 노란색
                cv2.rectangle(img, (x, y), (x+bw, y+bh), (0, 255, 0), 1)

            with mgr.lock:
                mgr.latest_results = objs
                mgr.stats["fps"] = round(1.0 / (time.time() - fps_time + 1e-6), 1)
                fps_time = time.time()
                if mgr.display_queue.full():
                    try: mgr.display_queue.get_nowait()
                    except: pass
                mgr.display_queue.put(img)
            
            process_decision(objs)
        except Empty: continue
        except Exception as e: logger.error(f"Inference Error: {e}")

def process_decision(objs):
    if not objs or mgr.locked_target: return
    best = min(objs, key=lambda o: math.sqrt((o['center'][0]-320)**2 + (o['center'][1]-480)**2))
    with mgr.lock:
        mgr.vote_buffer.append(best['class'])
        if len(mgr.vote_buffer) >= 7 and Counter(mgr.vote_buffer).most_common(1)[0][1] >= 4:
            mgr.locked_target = best
            mapping = {"S_Normal":1, "M_Normal":2, "B_Normal":3, "S_Abnormal":4, "M_Abnormal":5, "B_Abnormal":6}
            if mgr.ser: mgr.ser.write(f"SORT:{mapping.get(best['class'], 0)}\n".encode())
            mgr.detection_log.append({"time": datetime.now().strftime('%H:%M:%S'), "class": best['class'], "conf": best['conf']})
            if "Normal" in best['class']: mgr.stats["good"] += 1
            else: mgr.stats["defect"] += 1
            threading.Timer(3.0, lambda: setattr(mgr, 'locked_target', None)).start()

# ✅ [6] FastAPI 대시보드 서버
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.add_middleware(CORSMiddleware, allow_origins=["*"])
# API 키는 환경변수로 받는다 -- 코드에 키를 직접 적으면 저장소를 공개하는
# 순간 그대로 노출된다. 실행 전에 GEMINI_API_KEY 를 설정할 것.
#   (Windows) set GEMINI_API_KEY=발급받은_키
#   (Linux)   export GEMINI_API_KEY=발급받은_키
gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})

@app.get("/video_feed")
async def video_feed():
    def gen():
        while True:
            try:
                img = mgr.display_queue.get(timeout=1)
                _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            except Empty: continue
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/data")
async def get_data():
    with mgr.lock: return {"stats": mgr.stats, "detection_log": list(mgr.detection_log)}

@app.post("/ask_gemini")
async def ask_gemini(request: Request):
    try:
        data = await request.json()
        context = f"FPS:{mgr.stats['fps']}, 양품:{mgr.stats['good']}, 불량:{mgr.stats['defect']}. {data.get('prompt')}"
        res = gemini_client.models.generate_content(model="gemini-1.5-flash", contents=context)
        return {"answer": res.text}
    except Exception as e: return {"answer": f"AI Error: {e}"}

if __name__ == '__main__':
    ai_engine = TensorRTInferenceEngine('best.engine')
    mgr.init_arduino()
    
    # 산업용 병렬 스레드 가동
    threading.Thread(target=camera_worker, daemon=True).start()
    threading.Thread(target=inference_worker, args=(ai_engine,), daemon=True).start()
    threading.Thread(target=lambda: (exec("import psutil, time\nwhile True:\n with mgr.lock: mgr.stats['cpu']=psutil.cpu_percent(); mgr.stats['memory']=psutil.virtual_memory().percent\n time.sleep(1)")), daemon=True).start()
    
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)