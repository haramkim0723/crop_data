"""
yolo12m_v3: 작물(class 0) 오프라인 증강 후 학습
- augment_crop.py 실행해서 작물 전용 이미지 4배 증강 후 이 스크립트 실행
- v2 대비 변경: cls=0.5(복원), copy_paste=0.0(제거), close_mosaic=10(복원), cos_lr=True(추가)
"""
from ultralytics import YOLO

model = YOLO("/mnt/c/Users/pc/Downloads/sugar_beet_yolo/weights/yolo12m.pt")
model.train(
    data="/mnt/c/Users/pc/Downloads/sugar_beet_yolo/detection/dataset.yaml",
    epochs=100,
    imgsz=1280,
    batch=4,
    patience=20,
    workers=8,
    device=0,
    project="/mnt/c/Users/pc/Downloads/sugar_beet_yolo/detection/runs",
    name="yolo12m_v3",
    exist_ok=True,
    verbose=True,

    cos_lr=True,        # cosine LR schedule 추가
    close_mosaic=10,    # 기본값 복원 (v2에서 20으로 올렸다가 복원)
    cls=0.5,            # 기본값 복원 (v2에서 1.0으로 올렸다가 복원)

    hsv_s=0.9,          # 채도 변화 강화 (기본 0.7) → 조명/색상 다양성
    hsv_v=0.6,          # 명도 변화 강화 (기본 0.4) → 그림자/밝기 다양성
)
