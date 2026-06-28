import onnxruntime as ort
import numpy as np
import cv2

session = ort.InferenceSession("/home/user/my_pi_project/yolov5/best.onnx")
input_name = session.get_inputs()[0].name
output_names = [o.name for o in session.get_outputs()]

print("✅ input name:", input_name)
print("✅ output names:", output_names)

# 임의의 더미 이미지
img = np.zeros((1, 3, 640, 640), dtype=np.float32)

out = session.run(None, {input_name: img})

print("✅ output count:", len(out))
for i, o in enumerate(out):
    print(f"[{i}] output shape = {np.array(o).shape}")
