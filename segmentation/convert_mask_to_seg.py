"""
colorCleaned PNG 마스크 → YOLO segmentation 포맷(.txt 폴리곤)으로 변환.

색상 매핑 (BGR):
  (0, 255, 0) = 작물 → class 0
  (0, 0, 255) = 잡초 → class 1
  (0, 0, 0)   = 배경 → 무시

출력 구조:
  seg_dataset/
    train/images/  (기존 이미지 복사)
    train/labels/  (변환된 폴리곤 .txt)
    val/images/
    val/labels/
    test/images/
    test/labels/
    dataset.yaml
"""
import cv2
import numpy as np
import shutil
from pathlib import Path

MASK_DIR   = Path("/mnt/c/Users/pc/Downloads/ijrr_download_scripts/ijrr_download_scripts/crop_weed_annotations/colorCleaned")
OUT_DIR    = Path("/mnt/c/Users/pc/Downloads/sugar_beet_yolo/segmentation")

# detection 폴더의 기존 split
DETECTION  = Path("/mnt/c/Users/pc/Downloads/sugar_beet_yolo/detection")
SPLITS = {
    "train": DETECTION / "train",
    "val":   DETECTION / "val",
    "test":  DETECTION / "test",
}

# 색상(BGR) → 클래스
COLOR_MAP = {
    (0, 255, 0): 0,  # 작물
    (0, 0, 255): 1,  # 잡초
}

MIN_AREA = 50  # 너무 작은 컨투어 무시 (노이즈 제거)
EPSILON_RATIO = 0.002  # 폴리곤 단순화 비율


def mask_to_polygons(mask_path, img_w, img_h):
    """마스크 이미지 → [(class_id, [(x,y), ...]), ...]"""
    mask = cv2.imread(str(mask_path))
    if mask is None:
        return []

    results = []
    for bgr, cls_id in COLOR_MAP.items():
        # 해당 색상 픽셀만 추출
        color_mask = cv2.inRange(mask, np.array(bgr), np.array(bgr))
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_AREA:
                continue
            # 폴리곤 단순화
            epsilon = EPSILON_RATIO * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) < 3:
                continue
            # 정규화 좌표
            points = [(p[0][0] / img_w, p[0][1] / img_h) for p in approx]
            results.append((cls_id, points))

    return results


def save_yolo_seg(label_path, polygons):
    """YOLO seg 포맷으로 저장: class x1 y1 x2 y2 ..."""
    lines = []
    for cls_id, points in polygons:
        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in points)
        lines.append(f"{cls_id} {coords}")
    label_path.write_text("\n".join(lines))


def process_split(split_name, src_dir):
    img_dir   = src_dir / "images"
    out_img   = OUT_DIR / split_name / "images"
    out_label = OUT_DIR / split_name / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_label.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(img_dir.glob("*.png"))
    # 증강 파일 제외 (원본만)
    img_paths = [p for p in img_paths
                 if "_weedaug" not in p.stem
                 and "_cropaug" not in p.stem
                 and "_hardneg" not in p.stem]

    ok, skip = 0, 0
    for img_path in img_paths:
        mask_path = MASK_DIR / img_path.name
        if not mask_path.exists():
            skip += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            skip += 1
            continue
        h, w = img.shape[:2]

        polygons = mask_to_polygons(mask_path, w, h)

        # 이미지 복사
        shutil.copy(img_path, out_img / img_path.name)

        # 라벨 저장 (객체 없으면 빈 파일)
        save_yolo_seg(out_label / (img_path.stem + ".txt"), polygons)
        ok += 1

    print(f"  [{split_name}] 완료: {ok}장 / 마스크 없음: {skip}장")
    return ok


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("마스크 → YOLO seg 변환 시작\n")

    total = 0
    for split_name, src_dir in SPLITS.items():
        total += process_split(split_name, src_dir)

    # dataset.yaml 생성
    yaml_path = OUT_DIR / "dataset.yaml"
    yaml_path.write_text(
        f"path: {OUT_DIR.resolve().as_posix()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n\n"
        "nc: 2\n"
        "names: ['작물', '잡초']\n"
    )

    print(f"\n총 {total}장 변환 완료")
    print(f"dataset.yaml 생성: {yaml_path}")
    print("\n학습 명령:")
    print("  yolo segment train model=yolo12m-seg.pt "
          f"data={yaml_path.resolve().as_posix()} "
          "epochs=100 imgsz=1280 batch=4")


if __name__ == "__main__":
    main()
