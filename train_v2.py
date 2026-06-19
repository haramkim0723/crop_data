import subprocess
import sys
import os

DATASET_YAML = "C:/Users/pc/Downloads/sugar_beet_yolo/dataset.yaml"
PROJECT      = "C:/Users/pc/Downloads/sugar_beet_yolo/runs"
EPOCHS       = 100
IMGSZ        = 1280
PATIENCE     = 20
WORKERS      = 8

experiments = [
    {"model": "yolo11m.pt", "name": "yolo11m_v2", "device": 0},
    {"model": "yolo12m.pt", "name": "yolo12m_v2", "device": 1},
]

procs = []

for exp in experiments:
    cmd = [
        sys.executable, "-c",
        f"""
from ultralytics import YOLO
model = YOLO("{exp['model']}")
model.train(
    data="{DATASET_YAML}",
    epochs={EPOCHS},
    imgsz={IMGSZ},
    batch=-1,
    patience={PATIENCE},
    workers={WORKERS},
    device={exp['device']},
    project="{PROJECT}",
    name="{exp['name']}",
    exist_ok=True,
    verbose=True,

    # 클래스 불균형 개선
    cls=1.0,           # 분류 손실 가중치 2배 (기본 0.5) → 작물/잡초 구분 강화
    copy_paste=0.3,    # 잡초 인스턴스 복붙 증강 → 소수 클래스 보완

    # 탑뷰 촬영 특성 반영
    flipud=0.5,        # 상하반전 (탑뷰라 유효)
    degrees=15.0,      # 회전 증강 (탑뷰라 어느 방향이든 유효)

    # 기존 augmentation 유지
    close_mosaic=20,   # 모자이크를 더 오래 유지
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    fliplr=0.5,
)
"""
    ]

    log_path = f"C:/Users/pc/Downloads/sugar_beet_yolo/runs/{exp['name']}_train.log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    print(f"[START] {exp['name']} on GPU {exp['device']}")
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
    procs.append((exp['name'], proc, log_path))

print(f"\n2개 모델 동시 학습 시작!")
print(f"로그 확인:")
for name, _, log in procs:
    print(f"  {name}: {log}")
print()

for name, proc, log in procs:
    proc.wait()
    code = proc.returncode
    status = "완료" if code == 0 else f"실패 (exit {code})"
    print(f"[{status}] {name}")

print("\n=== 전체 학습 완료 ===")
print(f"결과 경로: {PROJECT}")
