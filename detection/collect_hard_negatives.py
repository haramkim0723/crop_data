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


def has_intersection(b1, b2):
    """두 박스가 조금이라도 겹치는지 확인 (YOLO xywh 정규화)"""
    def to_xyxy(b):
        cx, cy, w, h = b
        return cx - w/2, cy - h/2, cx + w/2, cy + h/2

    x1, y1, x2, y2 = to_xyxy(b1)
    x1g, y1g, x2g, y2g = to_xyxy(b2)

    return x1 < x2g and x2 > x1g and y1 < y2g and y2 > y1g


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
    """GT 박스와 조금이라도 겹치면 제외 → 완전히 빈 배경 FP만 수집"""
    for gt in gt_boxes:
        if has_intersection(pred_box, gt[1:]):
            return False
    return True


def expand_to_max(cx, cy, w, h, gt_boxes):
    """FP 박스를 GT 박스에 닿기 직전까지 사방으로 최대 확장 (정규화 좌표)"""
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    # 이미지 경계까지 최대 확장 후 GT와 겹치면 경계 후퇴
    left  = 0.0
    top   = 0.0
    right = 1.0
    bot   = 1.0

    for gt in gt_boxes:
        _, gcx, gcy, gw, gh = gt
        gx1, gy1 = gcx - gw / 2, gcy - gh / 2
        gx2, gy2 = gcx + gw / 2, gcy + gh / 2

        # FP 박스와 GT 박스가 수직으로 겹치는 경우에만 좌우 경계 조정
        if gy1 < y2 and gy2 > y1:
            if gx2 <= x1:  # GT가 FP 왼쪽
                left = max(left, gx2)
            if gx1 >= x2:  # GT가 FP 오른쪽
                right = min(right, gx1)

        # FP 박스와 GT 박스가 수평으로 겹치는 경우에만 상하 경계 조정
        if gx1 < x2 and gx2 > x1:
            if gy2 <= y1:  # GT가 FP 위쪽
                top = max(top, gy2)
            if gy1 >= y2:  # GT가 FP 아래쪽
                bot = min(bot, gy1)

    return left, top, right, bot


def crop_patch(img, x1, y1, x2, y2):
    """정규화 좌표로 이미지 크롭"""
    H, W = img.shape[:2]
    px1 = max(0, int(x1 * W))
    py1 = max(0, int(y1 * H))
    px2 = min(W, int(x2 * W))
    py2 = min(H, int(y2 * H))
    return img[py1:py2, px1:px2]


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
            x1, y1, x2, y2 = expand_to_max(cx, cy, w, h, gt_boxes)
            patch = crop_patch(img, x1, y1, x2, y2)
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
