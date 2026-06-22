"""
현재 모델(yolo12m/best.pt)로 val 이미지에 inference 후,
배경을 작물(class 0)로 잘못 예측한 FP 박스 영역만 잘라서 train에 추가.

- FP 박스 영역만 패치로 크롭
- 라벨은 빈 파일 (이 패치엔 아무것도 없다고 학습)
- 잡초 GT와 겹치는 예측은 제외 (순수 배경 FP만)
"""
from ultralytics import YOLO
from pathlib import Path
import cv2

MODEL_PATH  = "runs/yolo12m/weights/best.pt"
IMG_DIR     = Path("val/images")
LABEL_DIR   = Path("val/labels")
TRAIN_IMG   = Path("train/images")
TRAIN_LABEL = Path("train/labels")
SUFFIX      = "_hardneg"

CONF        = 0.25  # inference confidence threshold
IOU_THRESH  = 0.3   # GT와 이 IoU 이상 겹치면 FP 아님
CROP_CLASS  = 0     # 작물 클래스
PADDING     = 0.02  # 패치 크롭 시 박스 주변 여백 (정규화 비율)


def box_iou(b1, b2):
    """YOLO xywh(정규화) → IoU 계산"""
    def to_xyxy(b):
        cx, cy, w, h = b
        return cx - w/2, cy - h/2, cx + w/2, cy + h/2

    x1, y1, x2, y2 = to_xyxy(b1)
    x1g, y1g, x2g, y2g = to_xyxy(b2)

    xi1, yi1 = max(x1, x1g), max(y1, y1g)
    xi2, yi2 = min(x2, x2g), min(y2, y2g)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter == 0:
        return 0.0

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2g - x1g) * (y2g - y1g)
    return inter / (area1 + area2 - inter)


def load_gt(label_path):
    """GT 라벨 로드 → [(cls, cx, cy, w, h), ...]"""
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().strip().split('\n'):
        if line.strip():
            parts = line.split()
            boxes.append((int(parts[0]), float(parts[1]), float(parts[2]),
                          float(parts[3]), float(parts[4])))
    return boxes


def is_fp(pred_box, gt_boxes):
    """GT 박스(crop/weed 무관)와 충분히 안 겹치면 순수 배경 FP"""
    for gt in gt_boxes:
        if box_iou(pred_box, gt[1:]) >= IOU_THRESH:
            return False
    return True


def crop_patch(img, cx, cy, w, h, padding=PADDING):
    """FP 박스 영역을 패딩 포함해서 크롭"""
    H, W = img.shape[:2]
    x1 = max(0, int((cx - w/2 - padding) * W))
    y1 = max(0, int((cy - h/2 - padding) * H))
    x2 = min(W, int((cx + w/2 + padding) * W))
    y2 = min(H, int((cy + h/2 + padding) * H))
    return img[y1:y2, x1:x2]


def main():
    model = YOLO(MODEL_PATH)
    img_paths = sorted(IMG_DIR.glob("*.png"))
    print(f"대상 이미지: {len(img_paths)}장")

    added = 0
    for img_path in img_paths:
        results = model(str(img_path), conf=CONF, verbose=False)
        preds = results[0].boxes

        if preds is None or len(preds) == 0:
            continue

        # 작물(class 0)로 예측한 박스
        crop_preds = [
            tuple(box.xywhn[0].tolist())
            for box in preds
            if int(box.cls[0]) == CROP_CLASS
        ]
        if not crop_preds:
            continue

        gt_boxes = load_gt(LABEL_DIR / (img_path.stem + ".txt"))
        fp_boxes = [p for p in crop_preds if is_fp(p, gt_boxes)]

        if not fp_boxes:
            continue

        img = cv2.imread(str(img_path))

        for i, (cx, cy, w, h) in enumerate(fp_boxes):
            patch = crop_patch(img, cx, cy, w, h)
            if patch.size == 0:
                continue

            stem = f"{img_path.stem}{SUFFIX}_{i}"
            # 이미 있으면 스킵
            if (TRAIN_IMG / f"{stem}.png").exists():
                continue

            cv2.imwrite(str(TRAIN_IMG / f"{stem}.png"), patch)
            (TRAIN_LABEL / f"{stem}.txt").write_text("")  # 빈 라벨
            added += 1

        if added % 100 == 0 and added > 0:
            print(f"  수집 중... {added}장")

    print(f"\nhard negative 패치 추가 완료: {added}장")
    print(f"삭제하려면: python collect_hard_negatives.py --cleanup")


def cleanup():
    removed = 0
    for p in list(TRAIN_IMG.glob(f"*{SUFFIX}*.png")) + \
              list(TRAIN_LABEL.glob(f"*{SUFFIX}*.txt")):
        p.unlink()
        removed += 1
    print(f"hard negative {removed}개 삭제 완료")


if __name__ == "__main__":
    import sys
    if "--cleanup" in sys.argv:
        cleanup()
    else:
        main()
