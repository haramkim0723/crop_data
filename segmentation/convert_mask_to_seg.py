"""
colorCleaned PNG 마스크 → YOLO segmentation 포맷(.txt 폴리곤)으로 변환.

소스:
  1. top-level colorCleaned (7034장) + annotations/rgb/
  2. CKA 폴더 전용 마스크 (4518장) + CKA/images/rgb/ 또는 annotations/rgb/

색상 매핑 (BGR):
  (0, 255, 0) = 작물 → class 0
  (0, 0, 255) = 잡초 → class 1
  (0, 0, 0)   = 배경 → 무시

출력:
  segmentation/train|val|test/images + labels
  segmentation/dataset.yaml

Split: train 80% / val 15% / test 5%
"""
import cv2
import numpy as np
import shutil
import random
from pathlib import Path

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
ANNOT_BASE = Path("/mnt/c/Users/pc/Downloads/ijrr_download_scripts/ijrr_download_scripts/crop_weed_annotations")
TOP_MASK   = ANNOT_BASE / "colorCleaned"
TOP_RGB    = ANNOT_BASE / "rgb"
CKA_BASE   = ANNOT_BASE / "ijrr_sugarbeets_2016_annotations"
OUT_BASE   = Path("/mnt/c/Users/pc/Downloads/sugar_beet_yolo/segmentation")

# ── 파라미터 ──────────────────────────────────────────────────────────────────
COLOR_MAP     = {(0, 255, 0): 0, (0, 0, 255): 1}  # BGR → class
MIN_AREA      = 50
EPSILON_RATIO = 0.002
SPLIT_RATIO   = (0.80, 0.15, 0.05)  # train / val / test
SEED          = 42


# ── 변환 함수 ─────────────────────────────────────────────────────────────────
def mask_to_polygons(mask_path, img_w, img_h):
    mask = cv2.imread(str(mask_path))
    if mask is None:
        return []
    results = []
    for bgr, cls_id in COLOR_MAP.items():
        color_mask = cv2.inRange(mask, np.array(bgr), np.array(bgr))
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_AREA:
                continue
            eps    = EPSILON_RATIO * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, eps, True)
            if len(approx) < 3:
                continue
            points = [(p[0][0] / img_w, p[0][1] / img_h) for p in approx]
            results.append((cls_id, points))
    return results


def save_yolo_seg(label_path, polygons):
    lines = []
    for cls_id, points in polygons:
        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in points)
        lines.append(f"{cls_id} {coords}")
    label_path.write_text("\n".join(lines))


# ── 소스 수집 ─────────────────────────────────────────────────────────────────
def collect_pairs():
    """(mask_path, rgb_path) 쌍 수집 (중복 제거)"""
    pairs = {}  # filename → (mask_path, rgb_path)

    # 1. top-level colorCleaned
    for mask in TOP_MASK.glob("*.png"):
        rgb = TOP_RGB / mask.name
        if rgb.exists():
            pairs[mask.name] = (mask, rgb)

    # 2. CKA 전용 마스크
    top_names = set(pairs.keys())
    for cka in CKA_BASE.iterdir():
        mask_dir = cka / "annotations/dlp/colorCleaned"
        rgb_cka  = cka / "images/rgb"
        if not mask_dir.exists():
            continue
        for mask in mask_dir.glob("*.png"):
            if mask.name in top_names:
                continue  # top-level에 이미 있음
            # RGB 탐색: CKA rgb → top rgb 순
            rgb = rgb_cka / mask.name
            if not rgb.exists():
                rgb = TOP_RGB / mask.name
            if rgb.exists():
                pairs[mask.name] = (mask, rgb)

    return list(pairs.values())


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    print("소스 이미지 수집 중...")
    pairs = collect_pairs()
    print(f"총 {len(pairs)}쌍 수집 완료\n")

    # 랜덤 split
    random.seed(SEED)
    random.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * SPLIT_RATIO[0])
    n_val   = int(n * SPLIT_RATIO[1])
    splits  = {
        "train": pairs[:n_train],
        "val":   pairs[n_train:n_train + n_val],
        "test":  pairs[n_train + n_val:],
    }

    for split_name, split_pairs in splits.items():
        out_img   = OUT_BASE / split_name / "images"
        out_label = OUT_BASE / split_name / "labels"
        out_img.mkdir(parents=True, exist_ok=True)
        out_label.mkdir(parents=True, exist_ok=True)

        ok, skip = 0, 0
        for mask_path, rgb_path in split_pairs:
            img = cv2.imread(str(rgb_path))
            if img is None:
                skip += 1
                continue
            h, w = img.shape[:2]
            polygons = mask_to_polygons(mask_path, w, h)

            shutil.copy(rgb_path, out_img / rgb_path.name)
            save_yolo_seg(out_label / (rgb_path.stem + ".txt"), polygons)
            ok += 1

        print(f"  [{split_name}] {ok}장 완료 / {skip}장 스킵")

    # dataset.yaml
    yaml_path = OUT_BASE / "dataset.yaml"
    yaml_path.write_text(
        f"path: {OUT_BASE.as_posix()}\n"
        "train: train/images\n"
        "val:   val/images\n"
        "test:  test/images\n\n"
        "nc: 2\n"
        "names: ['작물', '잡초']\n"
    )

    total = sum(len(v) for v in splits.values())
    print(f"\n총 {total}장 변환 완료")
    print(f"  train: {len(splits['train'])}장")
    print(f"  val:   {len(splits['val'])}장")
    print(f"  test:  {len(splits['test'])}장")
    print(f"dataset.yaml: {yaml_path}")


if __name__ == "__main__":
    main()
