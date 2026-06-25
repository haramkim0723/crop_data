"""
colorCleaned 마스크에서 객체가 전혀 없는 영역(순수 배경)을 crop해서
background negative 이미지로 추출.

방법:
  annotated 이미지 + colorCleaned 마스크를 읽어서
  PATCH_SIZE × PATCH_SIZE 윈도우를 랜덤 샘플링 → 해당 영역 마스크에
  green(작물) / red(잡초) 픽셀이 없으면 배경 패치로 저장.

출력: segmentation/train/images/ + 빈 labels/

실행:
  python extract_backgrounds.py           # 기본 500장
  python extract_backgrounds.py 1000      # 1000장
  python extract_backgrounds.py cleanup   # 추출 파일 삭제
"""

import sys
import cv2
import numpy as np
import random
from pathlib import Path

ANNOT_BASE = Path("C:/Users/pc/Downloads/ijrr_download_scripts/ijrr_download_scripts/crop_weed_annotations")
TOP_RGB    = ANNOT_BASE / "rgb"
TOP_MASK   = ANNOT_BASE / "colorCleaned"
CKA_BASE   = ANNOT_BASE / "ijrr_sugarbeets_2016_annotations"

TRAIN_IMG = Path("C:/Users/pc/Downloads/sugar_beet_yolo/detection/train/images")
TRAIN_LBL = Path("C:/Users/pc/Downloads/sugar_beet_yolo/detection/train/labels")

BG_PREFIX  = "bg_patch_"
PATCH_SIZE = 640        # 추출할 패치 크기 (px)
MAX_TRIES  = 50         # 이미지당 랜덤 시도 횟수
SEED       = 42
DEFAULT_N  = 500


# ── 소스 쌍 수집 ─────────────────────────────────────────────────────────────

def collect_annotated_pairs():
    """(rgb_path, mask_path) 쌍 수집 (마스크가 있는 이미지만)"""
    pairs = {}
    # top-level
    for mask in TOP_MASK.glob("*.png"):
        rgb = TOP_RGB / mask.name
        if rgb.exists():
            pairs[mask.name] = (rgb, mask)
    # CKA 서브폴더
    for cka_dir in CKA_BASE.iterdir():
        mask_dir = cka_dir / "annotations/dlp/colorCleaned"
        rgb_dir  = cka_dir / "images/rgb"
        if not mask_dir.exists():
            continue
        for mask in mask_dir.glob("*.png"):
            if mask.name in pairs:
                continue
            rgb = rgb_dir / mask.name
            if not rgb.exists():
                rgb = TOP_RGB / mask.name
            if rgb.exists():
                pairs[mask.name] = (rgb, mask)
    return list(pairs.values())


# ── 배경 패치 추출 ────────────────────────────────────────────────────────────

def has_annotation(mask_crop):
    """패치 영역에 crop(초록) 또는 weed(빨강) 픽셀이 있으면 True"""
    green = np.any(np.all(mask_crop == [0, 255, 0], axis=2))
    red   = np.any(np.all(mask_crop == [0, 0, 255], axis=2))
    return green or red


def extract_bg_patches(rgb_path, mask_path, rng):
    """이미지 1장에서 순수 배경 패치를 최대 1개 추출 → (patch_img, stem_name) or None"""
    img  = cv2.imread(str(rgb_path))
    mask = cv2.imread(str(mask_path))
    if img is None or mask is None:
        return None

    h, w = img.shape[:2]
    if h < PATCH_SIZE or w < PATCH_SIZE:
        return None

    for _ in range(MAX_TRIES):
        x = rng.randint(0, w - PATCH_SIZE)
        y = rng.randint(0, h - PATCH_SIZE)
        mask_crop = mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
        if not has_annotation(mask_crop):
            patch = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            name  = f"{BG_PREFIX}{rgb_path.stem}_x{x}_y{y}"
            return patch, name

    return None


# ── 메인 ─────────────────────────────────────────────────────────────────────

def extract(n: int):
    existing_names = {p.stem for p in TRAIN_IMG.glob("*.png")}

    print("annotated 이미지-마스크 쌍 수집 중...")
    pairs = collect_annotated_pairs()
    print(f"  쌍 수: {len(pairs):,}장")

    rng = random.Random(SEED)
    rng.shuffle(pairs)

    added = 0
    for rgb_path, mask_path in pairs:
        if added >= n:
            break
        result = extract_bg_patches(rgb_path, mask_path, rng)
        if result is None:
            continue
        patch, name = result
        if name in existing_names:
            continue

        out_img = TRAIN_IMG / f"{name}.png"
        out_lbl = TRAIN_LBL / f"{name}.txt"
        cv2.imwrite(str(out_img), patch)
        out_lbl.write_text("")   # 빈 라벨 = background negative
        existing_names.add(name)
        added += 1

        if added % 100 == 0:
            print(f"  진행: {added}/{n}")

    print(f"\n추출 완료: {added:,}장 → train/images/")
    print(f"이후 augment_v3.py 실행 시 bg aug에 포함됩니다.")


def cleanup():
    removed = 0
    for img_path in list(TRAIN_IMG.glob(f"{BG_PREFIX}*.png")):
        lbl_path = TRAIN_LBL / (img_path.stem + ".txt")
        img_path.unlink(missing_ok=True)
        lbl_path.unlink(missing_ok=True)
        removed += 1
    print(f"배경 패치 파일 {removed:,}개 삭제 완료")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "cleanup":
        cleanup()
    else:
        n = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_N
        extract(n)
