import torch

# best.pt 경로
model_path = "best.pt"

# 모델 로드
model = torch.load(model_path, map_location="cpu")  # CPU에서 테스트
model.eval()  # 추론 모드

# 더 나아가 YOLOv5/YOLOv8 hub 모델처럼 쓰려면
# from ultralytics import YOLO
# model = YOLO("best.pt")
