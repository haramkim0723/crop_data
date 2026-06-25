"""
배경 패치 추출 + 증강을 한 번에 실행.

단계:
  1. colorCleaned 마스크에서 객체 없는 영역을 640×640 패치로 추출 (기본 500장)
  2. 추출된 패치에 4가지 기하 변환 적용 → aug 파일 생성

출력: detection/train/images/ + detection/train/labels/ (빈 라벨)

실행:
  python detection/prepare_backgrounds.py           # 500장 추출 + 증강
  python detection/prepare_backgrounds.py 1000      # 1000장 추출 + 증강
  python detection/prepare_backgrounds.py cleanup   # 생성 파일 전부 삭제
"""

import sys
import cv2
import numpy as np
import random
from pathlib import Path

# ── 경로 ─────────────────────────────────────────────────────────────────────

ANNOT_BASE = Path("C:/Users/pc/Downloads/ijrr_download_scripts/ijrr_download_scripts/crop_weed_annotations")
TOP_RGB    = ANNOT_BASE / "rgb"
TOP_MASK   = ANNOT_BASE / "colorCleaned"
CKA_BASE   = ANNOT_BASE / "ijrr_sugarbeets_2016_annotations"

TRAIN_IMG  = Path("C:/Users/pc/Downloads/sugar_beet_yolo/detection/train/images")
TRAIN_LBL  = Path("C:/Users/pc/Downloads/sugar_beet_yolo/detection/train/labels")

BG_PREFIX  = "bg_patch_"
AUG_SUFFIX = "_bgaug"
PATCH_SIZE = 640
MAX_TRIES  = 50
SEED       = 42
DEFAULT_N  = 500

# ── 증강 변환 ─────────────────────────────────────────────────────────────────

TRANSFORMS = {
    "lr":    lambda img: cv2.flip(img, 1),
    "ud":    lambda img: cv2.flip(img, 0),
    "rot90": lambda img: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
    "rot270":lambda img: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
}

# ── 소스 수집 ─────────────────────────────────────────────────────────────────

def collect_annotated_pairs():
    pairs = {}
    for mask in TOP_MASK.glob("*.png"):
        rgb = TOP_RGB / mask.name
        if rgb.exists():
            pairs[mask.name] = (rgb, mask)
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

# ── 패치 추출 ─────────────────────────────────────────────────────────────────

def has_annotation(mask_crop):
    green = np.any(np.all(mask_crop == [0, 255, 0], axis=2))
    red   = np.any(np.all(mask_crop == [0, 0, 255], axis=2))
    return green or red


def try_extract_patch(rgb_path, mask_path, rng):
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
        if not has_annotation(mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]):
            name = f"{BG_PREFIX}{rgb_path.stem}_x{x}_y{y}"
            return img[y:y+PATCH_SIZE, x:x+PATCH_SIZE], name
    return None

# ── STEP 1: 추출 ─────────────────────────────────────────────────────────────

def step1_extract(n: int) -> list[str]:
    """배경 패치 추출. 추출된 stem 목록 반환."""
    existing = {p.stem for p in TRAIN_IMG.glob("*.png")}

    print(f"[1/2] 배경 패치 추출 (목표 {n}장)")
    pairs = collect_annotated_pairs()
    print(f"      소스 쌍: {len(pairs):,}장")

    rng = random.Random(SEED)
    rng.shuffle(pairs)

    added_stems = []
    for rgb_path, mask_path in pairs:
        if len(added_stems) >= n:
            break
        result = try_extract_patch(rgb_path, mask_path, rng)
        if result is None:
            continue
        patch, name = result
        if name in existing:
            continue
        cv2.imwrite(str(TRAIN_IMG / f"{name}.png"), patch)
        (TRAIN_LBL / f"{name}.txt").write_text("")
        existing.add(name)
        added_stems.append(name)
        if len(added_stems) % 100 == 0:
            print(f"      진행: {len(added_stems)}/{n}")

    print(f"      완료: {len(added_stems):,}장 추출")
    return added_stems

# ── STEP 2: 증강 ─────────────────────────────────────────────────────────────

def step2_augment(stems: list[str]):
    """추출된 패치에 4가지 변환 적용."""
    print(f"\n[2/2] 배경 패치 증강 ({len(stems):,}장 × {len(TRANSFORMS)}변환)")
    added = 0
    for stem in stems:
        src_img = TRAIN_IMG / f"{stem}.png"
        img = cv2.imread(str(src_img))
        if img is None:
            continue
        for mode, fn in TRANSFORMS.items():
            new_stem = f"{stem}{AUG_SUFFIX}_{mode}"
            out_img  = TRAIN_IMG / f"{new_stem}.png"
            if out_img.exists():
                continue
            cv2.imwrite(str(out_img), fn(img))
            (TRAIN_LBL / f"{new_stem}.txt").write_text("")
            added += 1

    print(f"      완료: {added:,}장 증강")

# ── 요약 ─────────────────────────────────────────────────────────────────────

def summary(n_extracted: int):
    total   = len(list(TRAIN_IMG.glob("*.png"))) + len(list(TRAIN_IMG.glob("*.jpg")))
    patches = len(list(TRAIN_IMG.glob(f"{BG_PREFIX}*.png")))
    aug     = len(list(TRAIN_IMG.glob(f"{BG_PREFIX}*{AUG_SUFFIX}*.png")))
    orig_bg = patches - aug
    print(f"\n결과 요약:")
    print(f"  배경 원본:  {orig_bg:,}장")
    print(f"  배경 aug:   {aug:,}장")
    print(f"  train 전체: {total:,}장")

# ── cleanup ───────────────────────────────────────────────────────────────────

def cleanup():
    removed = 0
    for img_path in list(TRAIN_IMG.glob(f"{BG_PREFIX}*.png")):
        lbl_path = TRAIN_LBL / (img_path.stem + ".txt")
        img_path.unlink(missing_ok=True)
        lbl_path.unlink(missing_ok=True)
        removed += 1
    print(f"배경 파일 {removed:,}개 삭제 완료")

# ── 진입점 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "cleanup":
        cleanup()
    else:
        n = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_N
        stems = step1_extract(n)
        step2_augment(stems)
        summary(n)
