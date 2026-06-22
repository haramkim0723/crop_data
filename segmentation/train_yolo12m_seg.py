from ultralytics import YOLO

model = YOLO("/mnt/c/Users/pc/Downloads/sugar_beet_yolo/weights/yolo12m.pt")
model.train(
    task="segment",
    data="/mnt/c/Users/pc/Downloads/sugar_beet_yolo/segmentation/dataset.yaml",
    epochs=100,
    imgsz=1280,
    batch=4,
    patience=20,
    workers=8,
    device=0,
    project="/mnt/c/Users/pc/Downloads/sugar_beet_yolo/segmentation/runs",
    name="yolo12m_seg_v1",
    exist_ok=True,
    verbose=True,
)
