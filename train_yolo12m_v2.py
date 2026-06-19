from ultralytics import YOLO

model = YOLO("yolo12m.pt")
model.train(
    data="/mnt/c/Users/pc/Downloads/sugar_beet_yolo/dataset.yaml",
    epochs=100,
    imgsz=1280,
    batch=4,
    patience=20,
    workers=8,
    device=1,
    project="/mnt/c/Users/pc/Downloads/sugar_beet_yolo/runs",
    name="yolo12m_v2",
    exist_ok=True,
    verbose=True,

    # 클래스 불균형 개선 (오프라인 증강 augment_weed.py로 이미 적용)
    cls=1.0,           # 분류 손실 가중치 2배 (기본 0.5) → 작물/잡초 구분 강화
    copy_paste=0.3,    # 인스턴스 복붙 증강

    # 기존 augmentation 유지 (flipud/degrees/fliplr는 오프라인에서 적용했으므로 제거)
    close_mosaic=20,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
)
