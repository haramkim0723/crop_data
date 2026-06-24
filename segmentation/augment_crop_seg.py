"""
seg 작물(class 0)만 포함된 이미지를 플립/회전 증강해서 train 폴더에 추가.
라벨 포맷: YOLO seg polygon (cls x1 y1 x2 y2 ... xn yn)
학습 후 cleanup() 호출해서 제거.
"""
import cv2
from pathlib import Path

IMG_DIR   = Path('/mnt/c/Users/pc/Downloads/sugar_beet_yolo/segmentation/train/images')
LABEL_DIR = Path('/mnt/c/Users/pc/Downloads/sugar_beet_yolo/segmentation/train/labels')
SUFFIX    = '_cropaug'


def load_seg_labels(label_path):
    """각 줄: cls x1 y1 x2 y2 ... → (cls, [(x,y), ...]) 리스트"""
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


def transform_points(points, mode):
    result = []
    for x, y in points:
        if mode == 'lr':       # 좌우반전
            result.append((1.0 - x, y))
        elif mode == 'ud':     # 상하반전
            result.append((x, 1.0 - y))
        elif mode == 'rot90':  # 90도 시계방향
            result.append((y, 1.0 - x))
        elif mode == 'rot270': # 270도 시계방향 (90도 반시계)
            result.append((1.0 - y, x))
    return result


def augment():
    # 작물(class 0)만 있는 이미지 찾기
    crop_only = []
    for lf in LABEL_DIR.glob('*.txt'):
        if SUFFIX in lf.stem:
            continue
        polygons = load_seg_labels(lf)
        classes = {cls for cls, _ in polygons}
        if classes == {0}:
            crop_only.append(lf.stem)

    print(f'작물 전용 이미지: {len(crop_only)}장')

    transforms = {
        'lr':    lambda img: cv2.flip(img, 1),
        'ud':    lambda img: cv2.flip(img, 0),
        'rot90': lambda img: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
        'rot270':lambda img: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
    }

    added = 0
    for stem in crop_only:
        img_path = IMG_DIR / f'{stem}.png'
        if not img_path.exists():
            img_path = IMG_DIR / f'{stem}.jpg'
        if not img_path.exists():
            continue

        lbl_path = LABEL_DIR / f'{stem}.txt'
        img      = cv2.imread(str(img_path))
        polygons = load_seg_labels(lbl_path)

        for mode, fn in transforms.items():
            new_stem     = f'{stem}{SUFFIX}_{mode}'
            new_img_path = img_path.parent / f'{new_stem}{img_path.suffix}'
            new_lbl_path = LABEL_DIR / f'{new_stem}.txt'

            if new_img_path.exists():
                continue

            aug_img = fn(img)
            aug_polygons = [(cls, transform_points(pts, mode)) for cls, pts in polygons]

            cv2.imwrite(str(new_img_path), aug_img)
            save_seg_labels(new_lbl_path, aug_polygons)
            added += 1

    print(f'증강 이미지 추가: {added}장')
    print(f'학습 후 cleanup() 실행해서 제거하세요.')


def cleanup():
    removed = 0
    for p in list(IMG_DIR.glob(f'*{SUFFIX}*')) + list(LABEL_DIR.glob(f'*{SUFFIX}*.txt')):
        p.unlink()
        removed += 1
    print(f'증강 파일 {removed}개 삭제 완료')


if __name__ == '__main__':
    augment()
