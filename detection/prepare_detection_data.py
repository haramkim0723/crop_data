"""
Detection 학습 데이터 전처리 올인원 스크립트.

단계:
  1. 작물 포함 이미지 aug  (목표 10,000장, 랜덤 샘플)
  2. 잡초 포함 이미지 aug  (목표 12,000장, 랜덤 샘플)
  3. 배경 패치 추출        (마스크에서 객체 없는 640×640 크롭, 기본 2,000장)

라벨 포맷: YOLO bbox  →  class x_center y_center width height

실행:
  python detection/prepare_detection_data.py           # 기본값 실행
  python detection/prepare_detection_data.py cleanup   # 생성 파일 전부 삭제
"""

import sys
import cv2
import numpy as np
import random
from pathlib import Path

# ── 경로 ─────────────────────────────────────────────────────────────────────

def _resolve(win_path: str) -> Path:
    """Windows 경로를 실행 환경(WSL/Git Bash/Windows)에 맞게 변환"""
    p = Path(win_path)
    if p.exists():
        return p
    # WSL: C:/... → /mnt/c/...
    wsl = Path("/mnt/c") / Path(*p.parts[1:])
    if wsl.exists():
        return wsl
    return p  # 존재 안 해도 원본 반환 (에러는 호출부에서 처리)


ANNOT_BASE = _resolve("C:/Users/pc/Downloads/ijrr_download_scripts/ijrr_download_scripts/crop_weed_annotations")
TOP_RGB    = ANNOT_BASE / "rgb"
TOP_MASK   = ANNOT_BASE / "colorCleaned"
CKA_BASE   = ANNOT_BASE / "ijrr_sugarbeets_2016_annotations"

TRAIN_IMG  = _resolve("C:/Users/pc/Downloads/sugar_beet_yolo/detection/train/images")
TRAIN_LBL  = _resolve("C:/Users/pc/Downloads/sugar_beet_yolo/detection/train/labels")

CROP_SUFFIX = "_cropaug"
WEED_SUFFIX = "_weedaug"
BG_PREFIX   = "bg_patch_"

PATCH_SIZE   = 640
MAX_TRIES    = 50
SEED         = 42
DEFAULT_BG_N = 2000
CROP_TARGET  = 10000
WEED_TARGET  = 12000


# ── 이미지 변환 ───────────────────────────────────────────────────────────────

def _img_lr(img):    return cv2.flip(img, 1)
def _img_ud(img):    return cv2.flip(img, 0)
def _img_r90(img):   return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
def _img_r270(img):  return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
def _img_r180(img):  return cv2.rotate(img, cv2.ROTATE_180)
def _img_bup(img):   return cv2.convertScaleAbs(img, alpha=1.3, beta=0)
def _img_bdn(img):   return cv2.convertScaleAbs(img, alpha=0.7, beta=0)
def _img_con(img):   return cv2.convertScaleAbs(img, alpha=1.4, beta=-30)


# ── YOLO bbox 변환 ────────────────────────────────────────────────────────────

def _bbox_lr(rows):
    return [[c, 1.0-x, y, w, h] for c, x, y, w, h in rows]

def _bbox_ud(rows):
    return [[c, x, 1.0-y, w, h] for c, x, y, w, h in rows]

def _bbox_r90(rows):
    return [[c, 1.0-y, x, h, w] for c, x, y, w, h in rows]

def _bbox_r270(rows):
    return [[c, y, 1.0-x, h, w] for c, x, y, w, h in rows]

def _bbox_r180(rows):
    return [[c, 1.0-x, 1.0-y, w, h] for c, x, y, w, h in rows]

def _bbox_id(rows):
    return rows


CROP_TRANSFORMS = {
    "lr":         (_img_lr,   _bbox_lr),
    "ud":         (_img_ud,   _bbox_ud),
    "rot90":      (_img_r90,  _bbox_r90),
    "rot270":     (_img_r270, _bbox_r270),
    "rot180":     (_img_r180, _bbox_r180),
    "bright_up":  (_img_bup,  _bbox_id),
    "bright_down":(_img_bdn,  _bbox_id),
    "contrast":   (_img_con,  _bbox_id),
}

WEED_TRANSFORMS = {k: CROP_TRANSFORMS[k] for k in ("lr", "ud", "rot90", "rot270")}


# ── 라벨 I/O ─────────────────────────────────────────────────────────────────

def load_bbox_labels(path: Path):
    text = path.read_text().strip()
    if not text:
        return []
    rows = []
    for line in text.split("\n"):
        parts = line.split()
        if len(parts) == 5:
            rows.append([int(parts[0])] + list(map(float, parts[1:])))
    return rows


def save_bbox_labels(path: Path, rows):
    lines = [f"{int(r[0])} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f}" for r in rows]
    path.write_text("\n".join(lines))


def find_img(stem: str) -> Path | None:
    for ext in (".png", ".jpg"):
        p = TRAIN_IMG / f"{stem}{ext}"
        if p.exists():
            return p
    return None


# ── STEP 1: 작물 aug ──────────────────────────────────────────────────────────

def step1_crop_aug():
    print(f"[1/3] 작물 포함 이미지 증강 (목표 {CROP_TARGET:,}장)")
    crop_stems = []
    for lf in TRAIN_LBL.glob("*.txt"):
        if any(s in lf.stem for s in (CROP_SUFFIX, WEED_SUFFIX, BG_PREFIX)):
            continue
        text = lf.read_text().strip()
        if not text:
            continue
        classes = set(int(l.split()[0]) for l in text.split("\n") if l.strip())
        if 0 in classes:
            crop_stems.append(lf.stem)

    print(f"      작물 포함 원본: {len(crop_stems):,}장")

    candidates = [(stem, mode) for stem in crop_stems for mode in CROP_TRANSFORMS]
    random.Random(SEED).shuffle(candidates)
    targets = candidates[:CROP_TARGET]

    added = 0
    for stem, mode in targets:
        img_path = find_img(stem)
        if img_path is None:
            continue
        img_fn, bbox_fn = CROP_TRANSFORMS[mode]
        new_stem = f"{stem}{CROP_SUFFIX}_{mode}"
        if (TRAIN_IMG / f"{new_stem}{img_path.suffix}").exists():
            continue
        img  = cv2.imread(str(img_path))
        rows = load_bbox_labels(TRAIN_LBL / f"{stem}.txt")
        cv2.imwrite(str(TRAIN_IMG / f"{new_stem}{img_path.suffix}"), img_fn(img))
        save_bbox_labels(TRAIN_LBL / f"{new_stem}.txt", bbox_fn(rows))
        added += 1

    print(f"      추가 완료: {added:,}장")


# ── STEP 2: 잡초 aug ──────────────────────────────────────────────────────────

def step2_weed_aug():
    print(f"\n[2/3] 잡초 포함 이미지 증강 (목표 {WEED_TARGET:,}장)")

    weed_stems = []
    for lf in TRAIN_LBL.glob("*.txt"):
        if any(s in lf.stem for s in (CROP_SUFFIX, WEED_SUFFIX, BG_PREFIX)):
            continue
        text = lf.read_text().strip()
        if not text:
            continue
        if any(int(l.split()[0]) == 1 for l in text.split("\n") if l.strip()):
            weed_stems.append(lf.stem)

    print(f"      잡초 포함 원본: {len(weed_stems):,}장")

    candidates = [(stem, mode) for stem in weed_stems for mode in WEED_TRANSFORMS]
    random.Random(SEED).shuffle(candidates)
    targets = candidates[:WEED_TARGET]

    added = 0
    for stem, mode in targets:
        img_path = find_img(stem)
        if img_path is None:
            continue
        img_fn, bbox_fn = WEED_TRANSFORMS[mode]
        new_stem = f"{stem}{WEED_SUFFIX}_{mode}"
        if (TRAIN_IMG / f"{new_stem}{img_path.suffix}").exists():
            continue
        img  = cv2.imread(str(img_path))
        rows = load_bbox_labels(TRAIN_LBL / f"{stem}.txt")
        cv2.imwrite(str(TRAIN_IMG / f"{new_stem}{img_path.suffix}"), img_fn(img))
        save_bbox_labels(TRAIN_LBL / f"{new_stem}.txt", bbox_fn(rows))
        added += 1

    print(f"      추가 완료: {added:,}장")


# ── STEP 3: 배경 패치 추출 ───────────────────────────────────────────────────

def collect_annotated_pairs():
    pairs = {}
    for mask in TOP_MASK.glob("*.png"):
        rgb = TOP_RGB / mask.name
        if rgb.exists():
            pairs[mask.name] = (rgb, mask)
    if not CKA_BASE.exists():
        return list(pairs.values())
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


def has_annotation(mask_crop):
    return (np.any(np.all(mask_crop == [0, 255, 0], axis=2)) or
            np.any(np.all(mask_crop == [0, 0, 255], axis=2)))


def step3_extract_bg(n: int):
    print(f"\n[3/3] 배경 패치 추출 (목표 {n}장)")
    existing = {p.stem for p in TRAIN_IMG.glob("*.png")}

    pairs = collect_annotated_pairs()
    print(f"      소스 쌍: {len(pairs):,}장")

    rng = random.Random(SEED)
    rng.shuffle(pairs)

    added_stems = []
    for rgb_path, mask_path in pairs:
        if len(added_stems) >= n:
            break
        img  = cv2.imread(str(rgb_path))
        mask = cv2.imread(str(mask_path))
        if img is None or mask is None:
            continue
        h, w = img.shape[:2]
        if h < PATCH_SIZE or w < PATCH_SIZE:
            continue
        for _ in range(MAX_TRIES):
            x = rng.randint(0, w - PATCH_SIZE)
            y = rng.randint(0, h - PATCH_SIZE)
            if not has_annotation(mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]):
                name = f"{BG_PREFIX}{rgb_path.stem}_x{x}_y{y}"
                if name in existing:
                    break
                cv2.imwrite(str(TRAIN_IMG / f"{name}.png"),
                            img[y:y+PATCH_SIZE, x:x+PATCH_SIZE])
                (TRAIN_LBL / f"{name}.txt").write_text("")
                existing.add(name)
                added_stems.append(name)
                break
        if len(added_stems) % 100 == 0 and added_stems:
            print(f"      진행: {len(added_stems)}/{n}")

    print(f"      추출 완료: {len(added_stems):,}장")


# ── 요약 / cleanup ────────────────────────────────────────────────────────────

def summary():
    total   = len(list(TRAIN_IMG.glob("*.png"))) + len(list(TRAIN_IMG.glob("*.jpg")))
    crop_a  = len(list(TRAIN_IMG.glob(f"*{CROP_SUFFIX}_*.png"))) + \
              len(list(TRAIN_IMG.glob(f"*{CROP_SUFFIX}_*.jpg")))
    weed_a  = len(list(TRAIN_IMG.glob(f"*{WEED_SUFFIX}_*.png"))) + \
              len(list(TRAIN_IMG.glob(f"*{WEED_SUFFIX}_*.jpg")))
    bg_orig = len(list(TRAIN_IMG.glob(f"{BG_PREFIX}*.png")))
    orig    = total - crop_a - weed_a - bg_orig
    print(f"\n최종 train 구성:")
    print(f"  원본:          {orig:,}장")
    print(f"  작물 aug:      {crop_a:,}장")
    print(f"  잡초 aug:      {weed_a:,}장")
    print(f"  배경:          {bg_orig:,}장")
    print(f"  ──────────────────")
    print(f"  합계:          {total:,}장")


def cleanup():
    removed = 0
    for suffix in (CROP_SUFFIX, WEED_SUFFIX):
        for p in list(TRAIN_IMG.glob(f"*{suffix}_*.png")) + \
                 list(TRAIN_IMG.glob(f"*{suffix}_*.jpg")) + \
                 list(TRAIN_LBL.glob(f"*{suffix}_*.txt")):
            p.unlink(missing_ok=True)
            removed += 1
    for p in list(TRAIN_IMG.glob(f"{BG_PREFIX}*.png")) + \
             list(TRAIN_LBL.glob(f"{BG_PREFIX}*.txt")):
        p.unlink(missing_ok=True)
        removed += 1
    print(f"생성 파일 {removed:,}개 삭제 완료")


# ── 진입점 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "cleanup":
        cleanup()
    else:
        bg_n = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_BG_N
        print("[0/3] 기존 aug 파일 정리")
        cleanup()
        print()
        step1_crop_aug()
        step2_weed_aug()
        step3_extract_bg(bg_n)
        summary()
