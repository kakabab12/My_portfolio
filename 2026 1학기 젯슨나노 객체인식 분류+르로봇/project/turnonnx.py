from ultralytics import YOLO

# 1. ONNX 모델 불러오기
model = YOLO("box.onnx")

# 2. TensorRT(.engine)로 변환
# half=True: 젯슨 GPU의 FP16 가속을 써서 속도를 2배 높입니다.
model.export(format="engine", device=0, half=True)



# from ultralytics import YOLO

# # 1. 학습한 모델 로드 (파일명 확인!)
# model = YOLO('box.pt')

# # 2. ONNX 포맷으로 변환
# # simplify=True 옵션을 주면 젯슨에서 TensorRT로 바꿀 때 훨씬 안정적입니다.
# success = model.export(format='onnx', simplify=True)

# print("✨ ONNX 변환 성공!" if success else "❌ 변환 실패")