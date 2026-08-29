import torch
import torch.nn as nn

m = torch.jit.load("best.torchscript")

detect_layer = None

# Detect-like layer 탐색
for module in m.modules():
    # Detect 레이어는 "m" attribute를 가지고 있다.
    if hasattr(module, 'm'):
        detect_layer = module

if detect_layer is None:
    print("❌ Detect layer not found.")
    exit()

print("✅ Detect layer found!")

# detect_layer.m 이 iterable이 아닐 수 있으므로 children() 사용
for idx, child in enumerate(detect_layer.children()):
    # Conv2d 레이어만 찾기
    if isinstance(child, nn.Conv2d):
        print(f"Conv {idx} weight shape:", child.weight.shape)

    # 경우에 따라 detect_layer.m 안에 또다른 Sequential 있을 수 있음
    for sub_idx, sub_child in enumerate(child.children()):
        if isinstance(sub_child, nn.Conv2d):
            print(f"Conv {idx}-{sub_idx} weight shape:", sub_child.weight.shape)

