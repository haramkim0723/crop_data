from ultralytics import YOLO

model = YOLO("yolo11m.pt")
model.train(
    data="/mnt/c/Users/pc/Downloads/sugar_beet_yolo/dataset.yaml",
    epochs=100,
    imgsz=1280,
    batch=8,
    patience=20,
    workers=8,
    device=2,
    project="/mnt/c/Users/pc/Downloads/sugar_beet_yolo/runs",
    name="yolo11m",
    exist_ok=True,
)
