"""
segmentation 대규모 증강 v3
목표: 증강 이미지 ~30,000장 추가 (원본 9,237 + 증강 ~30,000 = 총 ~39,000장)
  - 작물 전용(crop-only) aug: ~20,000장  (8가지 변환 × 2,547장)
  - 잡초 포함(weed)    aug: ~10,000장  (4가지 변환 중 랜덤 샘플)
  - 백그라운드(bg)      aug:    ~44장   (4가지 변환 × 11장)

실행:
  python augment_v3.py          # 증강 실행
  python augment_v3.py cleanup  # 생성 파일 전부 삭제
"""

import sys
import cv2
import random
import numpy as np
from pathlib import Path

IMG_DIR   = Path('C:/Users/pc/Downloads/sugar_beet_yolo/segmentation/train/images')
LABEL_DIR = Path('C:/Users/pc/Downloads/sugar_beet_yolo/segmentation/train/labels')

CROP_SUFFIX = '_cropaug'
WEED_SUFFIX = '_weedaug'
BG_SUFFIX   = '_bgaug'

CROP_AUG_TARGET = 20_000
WEED_AUG_TARGET = 10_000

SEED = 42


# ── 라벨 I/O ─────────────────────────────────────────────────────────────────

def load_labels(path: Path):
    """returns list of (cls_int, [(x,y), ...]) or [] if empty"""
    text = path.read_text().strip()
    if not text:
        return []
    rows = []
    for line in text.split('\n'):
        parts = line.split()
        if not parts:
            continue
        cls = int(parts[0])
        coords = list(map(float, parts[1:]))
        pts = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
        rows.append((cls, pts))
    return rows


def save_labels(path: Path, rows):
    lines = []
    for cls, pts in rows:
        coords = ' '.join(f'{x:.6f} {y:.6f}' for x, y in pts)
        lines.append(f'{cls} {coords}')
    path.write_text('\n'.join(lines))


# ── 변환 정의 ─────────────────────────────────────────────────────────────────

def _pts_lr(pts):   return [(1.0 - x, y) for x, y in pts]
def _pts_ud(pts):   return [(x, 1.0 - y) for x, y in pts]
def _pts_r90(pts):  return [(y, 1.0 - x) for x, y in pts]
def _pts_r270(pts): return [(1.0 - y, x) for x, y in pts]
def _pts_r180(pts): return [(1.0 - x, 1.0 - y) for x, y in pts]
def _pts_id(pts):   return pts


def _bright(img, alpha):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=0)


def _contrast(img, alpha=1.4, beta=-30):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def _saturation(img, scale=1.4):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# (img_fn, pts_fn) 쌍으로 정의
GEOM = {
    'lr':    (lambda i: cv2.flip(i, 1),                            _pts_lr),
    'ud':    (lambda i: cv2.flip(i, 0),                            _pts_ud),
    'rot90': (lambda i: cv2.rotate(i, cv2.ROTATE_90_CLOCKWISE),    _pts_r90),
    'rot270':(lambda i: cv2.rotate(i, cv2.ROTATE_90_COUNTERCLOCKWISE), _pts_r270),
    'rot180':(lambda i: cv2.rotate(i, cv2.ROTATE_180),             _pts_r180),
}

COLOR = {
    'bright_up':   (lambda i: _bright(i, 1.3),    _pts_id),
    'bright_down': (lambda i: _bright(i, 0.7),    _pts_id),
    'contrast_up': (lambda i: _contrast(i),       _pts_id),
    # saturation은 crop전용에만 (총 8가지 맞추기)
    'saturation':  (lambda i: _saturation(i),     _pts_id),
}

# crop 전용: 기하 5 + 색상 3 = 8가지 (saturation 제외)
CROP_TRANSFORMS = {
    **{k: GEOM[k] for k in ('lr', 'ud', 'rot90', 'rot270', 'rot180')},
    **{k: COLOR[k] for k in ('bright_up', 'bright_down', 'contrast_up')},
}

# weed / bg: 기하 4가지
WEED_TRANSFORMS = {k: GEOM[k] for k in ('lr', 'ud', 'rot90', 'rot270')}
BG_TRANSFORMS   = WEED_TRANSFORMS


# ── 유틸 ─────────────────────────────────────────────────────────────────────

def find_img(stem: str) -> Path | None:
    for ext in ('.png', '.jpg'):
        p = IMG_DIR / f'{stem}{ext}'
        if p.exists():
            return p
    return None


def apply_transform(img, rows, img_fn, pts_fn):
    new_img  = img_fn(img)
    new_rows = [(cls, pts_fn(pts)) for cls, pts in rows]
    return new_img, new_rows


def write_pair(img, rows, new_stem: str, orig_ext: str):
    img_path = IMG_DIR   / f'{new_stem}{orig_ext}'
    lbl_path = LABEL_DIR / f'{new_stem}.txt'
    cv2.imwrite(str(img_path), img)
    save_labels(lbl_path, rows)


# ── 카테고리 분류 ─────────────────────────────────────────────────────────────

def categorize():
    bg_stems, crop_stems, weed_stems = [], [], []
    for lf in LABEL_DIR.glob('*.txt'):
        if any(s in lf.stem for s in (CROP_SUFFIX, WEED_SUFFIX, BG_SUFFIX)):
            continue
        text = lf.read_text().strip()
        if not text:
            bg_stems.append(lf.stem)
            continue
        classes = set(int(l.split()[0]) for l in text.split('\n') if l.strip())
        if 1 in classes:
            weed_stems.append(lf.stem)
        else:
            crop_stems.append(lf.stem)
    return bg_stems, crop_stems, weed_stems


# ── 메인 증강 ─────────────────────────────────────────────────────────────────

def augment():
    random.seed(SEED)

    bg_stems, crop_stems, weed_stems = categorize()
    print(f'원본 분류 완료:')
    print(f'  백그라운드: {len(bg_stems)}장')
    print(f'  작물 전용:  {len(crop_stems)}장')
    print(f'  잡초 포함:  {len(weed_stems)}장')
    print()

    # ── 1. 작물 전용 aug (8가지 × crop_stems) ──────────────────────────────
    print(f'[1] 작물 aug 목표: {CROP_AUG_TARGET:,}장')
    crop_added = 0
    for stem in crop_stems:
        img_path = find_img(stem)
        if img_path is None:
            continue
        img  = cv2.imread(str(img_path))
        rows = load_labels(LABEL_DIR / f'{stem}.txt')

        for mode, (img_fn, pts_fn) in CROP_TRANSFORMS.items():
            new_stem = f'{stem}{CROP_SUFFIX}_{mode}'
            if (IMG_DIR / f'{new_stem}{img_path.suffix}').exists():
                continue
            new_img, new_rows = apply_transform(img, rows, img_fn, pts_fn)
            write_pair(new_img, new_rows, new_stem, img_path.suffix)
            crop_added += 1

    print(f'    추가 완료: {crop_added:,}장')

    # ── 2. 잡초 포함 aug (랜덤 샘플링으로 target 맞추기) ──────────────────
    print(f'\n[2] 잡초 aug 목표: {WEED_AUG_TARGET:,}장')
    # (stem, mode) 후보 전체 생성 후 셔플 → target만큼 적용
    weed_candidates = [
        (stem, mode)
        for stem in weed_stems
        for mode in WEED_TRANSFORMS
    ]
    random.shuffle(weed_candidates)
    targets = weed_candidates[:WEED_AUG_TARGET]

    weed_added = 0
    for stem, mode in targets:
        img_path = find_img(stem)
        if img_path is None:
            continue
        img_fn, pts_fn = WEED_TRANSFORMS[mode]
        new_stem = f'{stem}{WEED_SUFFIX}_{mode}'
        if (IMG_DIR / f'{new_stem}{img_path.suffix}').exists():
            continue
        img  = cv2.imread(str(img_path))
        rows = load_labels(LABEL_DIR / f'{stem}.txt')
        new_img, new_rows = apply_transform(img, rows, img_fn, pts_fn)
        write_pair(new_img, new_rows, new_stem, img_path.suffix)
        weed_added += 1

    print(f'    추가 완료: {weed_added:,}장')

    # ── 3. 백그라운드 aug (기하 5 × 색상 4 = 20가지 조합) ────────────────
    print(f'\n[3] 백그라운드 aug ({len(bg_stems)}장 × 20조합 = 최대 {len(bg_stems)*20}장)')
    bg_added = 0
    for stem in bg_stems:
        img_path = find_img(stem)
        if img_path is None:
            continue
        img = cv2.imread(str(img_path))
        for gk, ck in BG_TRANSFORM_COMBOS:
            geo_fn,   _ = GEOM[gk]
            color_fn, _ = COLOR[ck]
            new_stem = f'{stem}{BG_SUFFIX}_{gk}_{ck}'
            if (IMG_DIR / f'{new_stem}{img_path.suffix}').exists():
                continue
            new_img = color_fn(geo_fn(img))
            out_img  = IMG_DIR   / f'{new_stem}{img_path.suffix}'
            out_lbl  = LABEL_DIR / f'{new_stem}.txt'
            cv2.imwrite(str(out_img), new_img)
            out_lbl.write_text('')   # 빈 라벨 = background
            bg_added += 1

    print(f'    추가 완료: {bg_added:,}장')

    summary()


# ── 요약 / cleanup ────────────────────────────────────────────────────────────

def summary():
    total  = len(list(IMG_DIR.glob('*.png'))) + len(list(IMG_DIR.glob('*.jpg')))
    crop_a = len(list(IMG_DIR.glob(f'*{CROP_SUFFIX}_*.png')))
    weed_a = len(list(IMG_DIR.glob(f'*{WEED_SUFFIX}_*.png')))
    bg_a   = len(list(IMG_DIR.glob(f'*{BG_SUFFIX}_*.png')))
    orig   = total - crop_a - weed_a - bg_a
    print(f'\n최종 train 구성:')
    print(f'  원본:         {orig:,}장')
    print(f'  작물 aug:     {crop_a:,}장')
    print(f'  잡초 aug:     {weed_a:,}장')
    print(f'  백그라운드 aug:{bg_a:,}장')
    print(f'  ─────────────────')
    print(f'  합계:         {total:,}장')


def cleanup():
    removed = 0
    for suffix in (CROP_SUFFIX, WEED_SUFFIX, BG_SUFFIX):
        for p in list(IMG_DIR.glob(f'*{suffix}*')) + list(LABEL_DIR.glob(f'*{suffix}*.txt')):
            p.unlink(missing_ok=True)
            removed += 1
    print(f'증강 파일 {removed:,}개 삭제 완료')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'cleanup':
        cleanup()
    else:
        augment()
