from ultralytics import YOLO
import cv2

# 1. TensorRT 엔진 로드 (우리가 만든 보물!)
# 'best.engine' 파일이 같은 폴더에 있어야 합니다.
model = YOLO("best.engine", task="detect")

# 2. 웹캠 또는 영상 소스 연결
# 0번은 보통 젯슨에 연결된 USB 카메라나 CSI 카메라입니다.
cap = cv2.VideoCapture(0)

print("🚀 TensorRT 엔진 가동 시작! 상단 바의 GPU 점유율을 확인하세요.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 3. TensorRT로 초고속 추론 (device=0은 GPU 사용을 명시)
    results = model.predict(frame, device=0, conf=0.5, show=False)

    # 4. 화면에 결과 그리기
    annotated_frame = results[0].plot()
    cv2.imshow("Jetson Orin Nano - TensorRT Realtime", annotated_frame)

    # 'q' 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()