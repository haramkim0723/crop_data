from ultralytics import YOLO

model = YOLO("yolo12m.pt")
model.train(
    data="/mnt/c/Users/pc/Downloads/sugar_beet_yolo/dataset.yaml",
    epochs=100,
    imgsz=1280,
    batch=4,
    patience=20,
    workers=8,
    device=0,
    project="/mnt/c/Users/pc/Downloads/sugar_beet_yolo/runs",
    name="yolo12m",
    exist_ok=True,
)
