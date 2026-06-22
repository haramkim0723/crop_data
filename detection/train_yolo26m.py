from ultralytics import YOLO

model = YOLO("yolo26m.pt")
model.train(
    data="/mnt/c/Users/pc/Downloads/sugar_beet_yolo/dataset.yaml",
    epochs=100,
    imgsz=1280,
    batch=8,
    patience=20,
    workers=8,
    device=1,
    project="/mnt/c/Users/pc/Downloads/sugar_beet_yolo/runs",
    name="yolo26m",
    exist_ok=True,
)
