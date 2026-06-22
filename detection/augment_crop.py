"""
작물(class 0)만 포함된 이미지를 플립/회전 증강해서 train 폴더에 추가.
학습 후 cleanup() 호출해서 제거.
"""
import cv2
from pathlib import Path

IMG_DIR   = Path('train/images')
LABEL_DIR = Path('train/labels')
SUFFIX    = '_cropaug'


def flip_bbox(bboxes, mode):
    """YOLO 포맷 bbox 변환 (cx, cy, w, h 모두 0~1 정규화)"""
    result = []
    for b in bboxes:
        cls, cx, cy, w, h = b
        if mode == 'lr':    # 좌우반전
            cx = 1.0 - cx
        elif mode == 'ud':  # 상하반전
            cy = 1.0 - cy
        elif mode == 'rot90':  # 90도 회전
            cx, cy = cy, 1.0 - cx
            w, h   = h, w
        elif mode == 'rot270': # 270도 회전
            cx, cy = 1.0 - cy, cx
            w, h   = h, w
        result.append((cls, cx, cy, w, h))
    return result


def load_labels(label_path):
    lines = label_path.read_text().strip().split('\n')
    bboxes = []
    for line in lines:
        if line:
            parts = line.split()
            bboxes.append((int(parts[0]), float(parts[1]),
                           float(parts[2]), float(parts[3]), float(parts[4])))
    return bboxes


def save_labels(label_path, bboxes):
    lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
             for c, cx, cy, w, h in bboxes]
    label_path.write_text('\n'.join(lines))


def augment():
    # 작물만 있는 이미지 찾기
    crop_only = []
    for lf in LABEL_DIR.glob('*.txt'):
        classes = set()
        for line in lf.read_text().strip().split('\n'):
            if line:
                classes.add(int(line.split()[0]))
        if classes == {0}:  # 작물만
            crop_only.append(lf.stem)

    print(f"작물 전용 이미지: {len(crop_only)}장")

    transforms = {
        'lr':    lambda img: cv2.flip(img, 1),
        'ud':    lambda img: cv2.flip(img, 0),
        'rot90': lambda img: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
        'rot270':lambda img: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
    }

    added = 0
    for stem in crop_only:
        img_path = IMG_DIR / f"{stem}.png"
        lbl_path = LABEL_DIR / f"{stem}.txt"
        if not img_path.exists():
            continue

        img    = cv2.imread(str(img_path))
        bboxes = load_labels(lbl_path)

        for mode, fn in transforms.items():
            new_stem     = f"{stem}{SUFFIX}_{mode}"
            new_img_path = IMG_DIR   / f"{new_stem}.png"
            new_lbl_path = LABEL_DIR / f"{new_stem}.txt"

            if new_img_path.exists():
                continue

            aug_img    = fn(img)
            aug_bboxes = flip_bbox(bboxes, mode)

            cv2.imwrite(str(new_img_path), aug_img)
            save_labels(new_lbl_path, aug_bboxes)
            added += 1

    print(f"증강 이미지 추가: {added}장")
    print(f"학습 후 cleanup() 실행해서 제거하세요.")


def cleanup():
    removed = 0
    for p in list(IMG_DIR.glob(f'*{SUFFIX}*.png')) + list(LABEL_DIR.glob(f'*{SUFFIX}*.txt')):
        p.unlink()
        removed += 1
    print(f"증강 파일 {removed}개 삭제 완료")


if __name__ == '__main__':
    augment()