"""
yolo12m_seg_v2: detection v3 기법 적용
- yolo12m-seg.pt (seg 전용 가중치) 사용
- augment_crop_seg.py 실행해서 작물 전용 이미지 4배 증강 후 이 스크립트 실행
- v1 대비 변경: seg 전용 가중치, cos_lr=True, close_mosaic=10, cls=0.5
"""
from ultralytics import YOLO

model = YOLO("/mnt/c/Users/pc/Downloads/sugar_beet_yolo/weights/yolo11m-seg.pt")
model.train(
    task="segment",
    data="/mnt/c/Users/pc/Downloads/sugar_beet_yolo/segmentation/dataset.yaml",
    epochs=100,
    imgsz=1280,
    batch=8,
    patience=20,
    workers=8,
    device=2,
    project="/mnt/c/Users/pc/Downloads/sugar_beet_yolo/segmentation/runs",
    name="yolo12m_seg_v2",
    exist_ok=True,
    verbose=True,

    cos_lr=True,      # cosine LR schedule
    close_mosaic=10,  # 기본값
    cls=0.5,          # 기본값
)
