import subprocess
import sys
import os

# 학습 설정
DATASET_YAML = "/mnt/c/Users/pc/Downloads/sugar_beet_yolo/dataset.yaml"
PROJECT = "/mnt/c/Users/pc/Downloads/sugar_beet_yolo/runs"
EPOCHS = 100
IMGSZ = 1280
PATIENCE = 20
WORKERS = 8

experiments = [
    {"model": "yolo26s.pt", "name": "yolo26s", "device": 0},
    {"model": "yolo26m.pt", "name": "yolo26m", "device": 1},
    {"model": "yolo11m.pt", "name": "yolo11m", "device": 2},
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
)
"""
    ]
    log_path = f"/mnt/c/Users/pc/Downloads/sugar_beet_yolo/runs/{exp['name']}_train.log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    print(f"[START] {exp['name']} on GPU {exp['device']}")
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
    procs.append((exp['name'], proc, log_path))

print(f"\n3개 모델 동시 학습 시작!")
print(f"로그 확인:")
for name, _, log in procs:
    print(f"  {name}: {log}")
print()

# 모든 프로세스 완료 대기
for name, proc, log in procs:
    proc.wait()
    code = proc.returncode
    status = "완료" if code == 0 else f"실패 (exit {code})"
    print(f"[{status}] {name}")

print("\n=== 전체 학습 완료 ===")
print(f"결과 경로: {PROJECT}")
