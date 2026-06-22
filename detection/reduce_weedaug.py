"""
weedaug 파일을 절반으로 줄임 (랜덤 삭제)
"""
import random
from pathlib import Path

IMG_DIR   = Path('train/images')
LABEL_DIR = Path('train/labels')

random.seed(42)

weedaug_imgs = sorted(IMG_DIR.glob('*_weedaug*.png'))
print(f"현재 weedaug 이미지: {len(weedaug_imgs)}장")

to_delete = random.sample(weedaug_imgs, len(weedaug_imgs) // 2)

removed = 0
for img_path in to_delete:
    lbl_path = LABEL_DIR / (img_path.stem + '.txt')
    img_path.unlink()
    if lbl_path.exists():
        lbl_path.unlink()
    removed += 1

print(f"삭제 완료: {removed}장")
print(f"남은 weedaug 이미지: {len(weedaug_imgs) - removed}장")
