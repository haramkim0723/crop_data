"""
v3 best.pt로 val 이미지 예측 및 시각화
결과: detection/runs/v3_predict/
"""
from ultralytics import YOLO

model = YOLO("/mnt/c/Users/pc/Downloads/sugar_beet_yolo/detection/runs/yolo12m_v3/weights/best.pt")
model.predict(
    source="/mnt/c/Users/pc/Downloads/sugar_beet_yolo/detection/val/images",
    save=True,
    conf=0.25,
    iou=0.5,
    project="/mnt/c/Users/pc/Downloads/sugar_beet_yolo/detection/runs",
    name="v3_predict",
    exist_ok=True,
)
