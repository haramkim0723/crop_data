"""
증강 데이터 재균형:
  1. 작물 _cropaug 파일 절반 랜덤 제거
  2. 잡초 포함 이미지 ~3000장 증강 추가 (_weedaug suffix)
"""
import cv2
import random
from pathlib import Path

IMG_DIR   = Path('/mnt/c/Users/pc/Downloads/sugar_beet_yolo/segmentation/train/images')
LABEL_DIR = Path('/mnt/c/Users/pc/Downloads/sugar_beet_yolo/segmentation/train/labels')
SEED      = 42
WEED_TARGET = 3000

TRANSFORMS = {
    'lr':    lambda img: cv2.flip(img, 1),
    'ud':    lambda img: cv2.flip(img, 0),
    'rot90': lambda img: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
    'rot270':lambda img: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
}


def transform_points(points, mode):
    result = []
    for x, y in points:
        if mode == 'lr':
            result.append((1.0 - x, y))
        elif mode == 'ud':
            result.append((x, 1.0 - y))
        elif mode == 'rot90':
            result.append((y, 1.0 - x))
        elif mode == 'rot270':
            result.append((1.0 - y, x))
    return result


def load_seg_labels(label_path):
    results = []
    for line in label_path.read_text().strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cls = int(parts[0])
        coords = list(map(float, parts[1:]))
        points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
        results.append((cls, points))
    return results


def save_seg_labels(label_path, polygons):
    lines = []
    for cls, points in polygons:
        coords = ' '.join(f'{x:.6f} {y:.6f}' for x, y in points)
        lines.append(f'{cls} {coords}')
    label_path.write_text('\n'.join(lines))


# ── 1. 작물 aug 절반 제거 ─────────────────────────────────────────────────────
def trim_crop_aug():
    crop_aug_imgs = sorted(IMG_DIR.glob('*_cropaug_*.png'))
    random.seed(SEED)
    to_remove = random.sample(crop_aug_imgs, len(crop_aug_imgs) // 2)

    removed = 0
    for img_path in to_remove:
        lbl_path = LABEL_DIR / (img_path.stem + '.txt')
        img_path.unlink(missing_ok=True)
        lbl_path.unlink(missing_ok=True)
        removed += 1

    print(f'[1] 작물 aug 제거: {removed}장 삭제')
    remaining = len(list(IMG_DIR.glob('*_cropaug_*.png')))
    print(f'    남은 작물 aug: {remaining}장')


# ── 2. 잡초 포함 이미지 ~3000장 증강 ─────────────────────────────────────────
def augment_weed():
    # 잡초 포함 원본 이미지 찾기 (class 1 포함, _cropaug/_weedaug 제외)
    weed_stems = []
    for lf in LABEL_DIR.glob('*.txt'):
        if '_cropaug' in lf.stem or '_weedaug' in lf.stem:
            continue
        text = lf.read_text().strip()
        if not text:
            continue
        classes = set(int(l.split()[0]) for l in text.split('\n') if l.strip())
        if 1 in classes:
            weed_stems.append(lf.stem)

    print(f'\n[2] 잡초 포함 원본: {len(weed_stems)}장')

    # 각 이미지에 transform 1개씩 랜덤 적용해서 WEED_TARGET장 채우기
    random.seed(SEED)
    mode_list = list(TRANSFORMS.keys())

    # 목표 수만큼 (stem, mode) 조합 생성 (중복 없이)
    candidates = [(stem, mode) for stem in weed_stems for mode in mode_list]
    random.shuffle(candidates)
    targets = candidates[:WEED_TARGET]

    added = 0
    for stem, mode in targets:
        img_path = IMG_DIR / f'{stem}.png'
        if not img_path.exists():
            img_path = IMG_DIR / f'{stem}.jpg'
        if not img_path.exists():
            continue

        lbl_path    = LABEL_DIR / f'{stem}.txt'
        new_stem     = f'{stem}_weedaug_{mode}'
        new_img_path = img_path.parent / f'{new_stem}{img_path.suffix}'
        new_lbl_path = LABEL_DIR / f'{new_stem}.txt'

        if new_img_path.exists():
            continue

        img      = cv2.imread(str(img_path))
        polygons = load_seg_labels(lbl_path)

        aug_img      = TRANSFORMS[mode](img)
        aug_polygons = [(cls, transform_points(pts, mode)) for cls, pts in polygons]

        cv2.imwrite(str(new_img_path), aug_img)
        save_seg_labels(new_lbl_path, aug_polygons)
        added += 1

    print(f'    잡초 aug 추가: {added}장')


# ── 결과 요약 ─────────────────────────────────────────────────────────────────
def summary():
    total = len(list(IMG_DIR.glob('*.png'))) + len(list(IMG_DIR.glob('*.jpg')))
    crop_aug  = len(list(IMG_DIR.glob('*_cropaug_*.png')))
    weed_aug  = len(list(IMG_DIR.glob('*_weedaug_*.png')))
    orig = total - crop_aug - weed_aug
    print(f'\n최종 train 구성:')
    print(f'  원본:       {orig}장')
    print(f'  작물 aug:   {crop_aug}장')
    print(f'  잡초 aug:   {weed_aug}장')
    print(f'  합계:       {total}장')


if __name__ == '__main__':
    trim_crop_aug()
    augment_weed()
    summary()
