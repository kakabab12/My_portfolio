
from ultralytics import YOLO
import torch
print('GPU:', torch.cuda.is_available())
print('장치:', torch.cuda.get_device_name(0))
model = YOLO('/home/user/project/best.onnx', task='detect')
print('모델 로딩 OK')
"