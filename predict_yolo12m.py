from ultralytics import YOLO
from pathlib import Path
import time

MODEL_PATH = 'runs/yolo12m/weights/best.pt'
TEST_DIR   = 'test/images'
SAVE_DIR   = 'runs/yolo12m_predict'
IMGSZ      = 1280
CONF       = 0.4
DEVICE     = 'cuda:0'

model = YOLO(MODEL_PATH)
print(f"모델 클래스: {model.names}")
print(f"테스트 이미지: {len(list(Path(TEST_DIR).glob('*.png')))}장\n")

t0 = time.time()

results = model.predict(
    source=TEST_DIR,
    imgsz=IMGSZ,
    conf=CONF,
    device=DEVICE,
    save=True,
    save_txt=True,
    save_conf=True,
    project='runs',
    name='yolo12m_predict_conf40',
    exist_ok=True,
    verbose=False,
)

elapsed = time.time() - t0
total = len(results)
fps = total / elapsed

# 탐지 통계
class_counts = {}
no_detect = 0
for r in results:
    if len(r.boxes) == 0:
        no_detect += 1
    for cls_id in r.boxes.cls.tolist():
        cls_name = model.names[int(cls_id)]
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

print("===== 추론 완료 =====")
print(f"총 이미지: {total}장")
print(f"소요 시간: {elapsed:.1f}초 ({fps:.1f} FPS)")
print(f"미탐지 이미지: {no_detect}장")
print(f"\n[클래스별 탐지 수]")
for cls_name, count in sorted(class_counts.items()):
    print(f"  {cls_name}: {count}개")
print(f"\n결과 저장 위치: runs/yolo12m_predict_conf40/")
