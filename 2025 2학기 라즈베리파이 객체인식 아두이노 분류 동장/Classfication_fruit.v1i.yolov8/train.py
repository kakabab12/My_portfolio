from ultralytics import YOLO
import os

DATA_YAML = r"./data.yaml"

PROJECT = "runs_detect"
NAME = "nuts_detect_train"

EPOCHS = 50
IMG_SIZE = 640
BATCH = 16


def main():
    print("=== detection 학습 시작 ===")
    print("현재 작업 폴더:", os.getcwd())
    print("DATA_YAML 존재 여부:", os.path.exists(DATA_YAML))

    model = YOLO("yolov8n.pt")   # detection 모델

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        project=PROJECT,
        name=NAME,
        device=0,
        pretrained=True
    )

    best_path = os.path.join("runs", "detect", PROJECT, NAME, "weights", "best.pt")
    print("best.pt 경로:", best_path)

    best_model = YOLO(best_path)

    print("=== 검증 시작 ===")
    metrics = best_model.val()
    print("검증 결과:", metrics)

    print("=== ONNX 변환 시작 ===")
    export_path = best_model.export(
        format="onnx",
        imgsz=IMG_SIZE,
        opset=12,
        simplify=True
    )

    print("=== 완료 ===")
    print("ONNX 저장 경로:", export_path)


if __name__ == "__main__":
    main()