"""
현재 모델(yolo12m/best.pt)로 val 이미지 inference 후,
GT와 예측을 전체 클래스 기준으로 시각화해서 저장.

색상:
  GT 작물  → 초록 실선
  GT 잡초  → 노랑 실선
  예측 작물 → 파랑 점선
  예측 잡초 → 주황 점선

FN이 있는 이미지(GT는 있는데 예측 못 한 것)만 저장.
"""
from ultralytics import YOLO
from pathlib import Path
import cv2

MODEL_PATH = "runs/yolo12m/weights/best.pt"
IMG_DIR    = Path("val/images")
LABEL_DIR  = Path("val/labels")
OUT_DIR    = Path("fn_samples")

CONF       = 0.25
IOU_THRESH = 0.3
MAX_SAVE   = 200

CLASS_NAMES = {0: "작물", 1: "잡초"}

# GT 색상: 작물=초록, 잡초=노랑
GT_COLORS  = {0: (0, 255, 0), 1: (0, 255, 255)}
# 예측 색상: 작물=파랑, 잡초=주황
PRED_COLORS = {0: (255, 80, 0), 1: (0, 140, 255)}


def box_iou(b1, b2):
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
    return inter / ((x2-x1)*(y2-y1) + (x2g-x1g)*(y2g-y1g) - inter)


def load_gt(label_path):
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().strip().split('\n'):
        if line.strip():
            p = line.split()
            boxes.append((int(p[0]), float(p[1]), float(p[2]),
                          float(p[3]), float(p[4])))
    return boxes


def draw_box(vis, cx, cy, w, h, H, W, color, label, dashed=False):
    x1 = int((cx - w/2) * W); y1 = int((cy - h/2) * H)
    x2 = int((cx + w/2) * W); y2 = int((cy + h/2) * H)
    if dashed:
        # 점선 효과: 짧은 선분으로 분할
        dash = 12
        for i in range(x1, x2, dash*2):
            cv2.line(vis, (i, y1), (min(i+dash, x2), y1), color, 2)
            cv2.line(vis, (i, y2), (min(i+dash, x2), y2), color, 2)
        for i in range(y1, y2, dash*2):
            cv2.line(vis, (x1, i), (x1, min(i+dash, y2)), color, 2)
            cv2.line(vis, (x2, i), (x2, min(i+dash, y2)), color, 2)
    else:
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
    cv2.putText(vis, label, (x1, max(y1-5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def main():
    OUT_DIR.mkdir(exist_ok=True)

    model = YOLO(MODEL_PATH)
    img_paths = sorted(IMG_DIR.glob("*.png"))
    print(f"대상 이미지: {len(img_paths)}장")

    saved = 0
    total_fn = {0: 0, 1: 0}

    for img_path in img_paths:
        if saved >= MAX_SAVE:
            break

        gt_boxes = load_gt(LABEL_DIR / (img_path.stem + ".txt"))
        if not gt_boxes:
            continue

        results = model(str(img_path), conf=CONF, verbose=False)
        preds   = results[0].boxes

        # 예측 박스: {cls: [(cx,cy,w,h), ...]}
        pred_by_cls = {0: [], 1: []}
        if preds is not None:
            for box in preds:
                cls = int(box.cls[0])
                pred_by_cls[cls].append(tuple(box.xywhn[0].tolist()))

        # 클래스별로 FN 확인
        fn_boxes = []
        for gt in gt_boxes:
            cls = gt[0]
            same_cls_preds = pred_by_cls.get(cls, [])
            matched = any(box_iou(gt[1:], p) >= IOU_THRESH for p in same_cls_preds)
            if not matched:
                fn_boxes.append(gt)
                total_fn[cls] = total_fn.get(cls, 0) + 1

        if not fn_boxes:
            continue

        # 시각화
        img = cv2.imread(str(img_path))
        H, W = img.shape[:2]
        vis = img.copy()

        # GT 전체 그리기 (실선)
        for cls, cx, cy, w, h in gt_boxes:
            color = GT_COLORS.get(cls, (255, 255, 255))
            name  = CLASS_NAMES.get(cls, str(cls))
            is_fn = (cls, cx, cy, w, h) in fn_boxes
            label = f"GT {name}" + (" [FN]" if is_fn else "")
            thickness = 3 if is_fn else 1
            x1 = int((cx - w/2) * W); y1 = int((cy - h/2) * H)
            x2 = int((cx + w/2) * W); y2 = int((cy + h/2) * H)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(vis, label, (x1, max(y1-5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # 예측 전체 그리기 (점선)
        for cls, plist in pred_by_cls.items():
            color = PRED_COLORS.get(cls, (200, 200, 200))
            name  = CLASS_NAMES.get(cls, str(cls))
            for cx, cy, w, h in plist:
                draw_box(vis, cx, cy, w, h, H, W, color,
                         f"PRED {name}", dashed=True)

        out_name = f"fn_{saved:04d}_{img_path.name}"
        cv2.imwrite(str(OUT_DIR / out_name), vis)
        saved += 1

        if saved % 20 == 0:
            print(f"  저장 중... {saved}장")

    print(f"\n완료: {saved}장 저장")
    print(f"FN 누적 — 작물: {total_fn.get(0,0)}개 / 잡초: {total_fn.get(1,0)}개")
    print(f"확인 경로: {OUT_DIR.resolve()}")
    print()
    print("범례:")
    print("  초록 실선  = GT 작물  / 굵은 테두리 = FN(놓친 것)")
    print("  노랑 실선  = GT 잡초  / 굵은 테두리 = FN(놓친 것)")
    print("  파랑 점선  = 예측 작물")
    print("  주황 점선  = 예측 잡초")


if __name__ == "__main__":
    main()
