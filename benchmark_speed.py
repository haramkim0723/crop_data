import time
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

MODELS = {
    'yolo11m': 'runs/yolo11m/weights/best.pt',
    'yolo12m': 'runs/yolo12m/weights/best.pt',
    'yolo26m': 'runs/yolo26m/weights/best.pt',
}
COLORS = {'yolo11m': '#4C72B0', 'yolo12m': '#DD8452', 'yolo26m': '#55A868'}

TEST_DIR = Path('test/images')
IMGSZ = 1280
WARMUP = 10    # 워밍업 횟수
RUNS   = 50    # 측정 횟수

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"디바이스: {device}")

images = sorted(TEST_DIR.glob('*.png'))[:RUNS + WARMUP]
if len(images) < WARMUP + RUNS:
    images = (images * ((WARMUP + RUNS) // len(images) + 1))[:WARMUP + RUNS]

results_summary = {}

for name, path in MODELS.items():
    print(f"\n[{name}] 벤치마크 시작...")
    model = YOLO(path)

    # 워밍업
    for img in images[:WARMUP]:
        model.predict(str(img), imgsz=IMGSZ, device=device, verbose=False)

    # 측정
    times = []
    for img in images[WARMUP:WARMUP + RUNS]:
        t0 = time.perf_counter()
        model.predict(str(img), imgsz=IMGSZ, device=device, verbose=False)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    avg_ms  = np.mean(times)
    std_ms  = np.std(times)
    fps     = 1000 / avg_ms
    min_ms  = np.min(times)
    max_ms  = np.max(times)

    results_summary[name] = {
        'avg_ms': avg_ms, 'std_ms': std_ms,
        'fps': fps, 'min_ms': min_ms, 'max_ms': max_ms,
        'times': times
    }
    print(f"  평균: {avg_ms:.1f}ms | FPS: {fps:.1f} | std: {std_ms:.1f}ms")

# ── 결과 표 출력 ──────────────────────────────────────────────────────────────
print("\n===== 추론 속도 비교 =====")
print(f"{'모델':<10} {'평균(ms)':>10} {'FPS':>8} {'Min(ms)':>9} {'Max(ms)':>9} {'Std':>7}")
print("-" * 55)
for name, r in results_summary.items():
    print(f"{name:<10} {r['avg_ms']:>10.1f} {r['fps']:>8.1f} {r['min_ms']:>9.1f} {r['max_ms']:>9.1f} {r['std_ms']:>7.1f}")

# ── 그래프 ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f'추론 속도 비교 (device: {device}, imgsz: {IMGSZ})', fontsize=13, fontweight='bold')

names = list(results_summary.keys())
colors = [COLORS[n] for n in names]

# 1) FPS 막대
ax = axes[0]
fps_vals = [results_summary[n]['fps'] for n in names]
bars = ax.bar(names, fps_vals, color=colors, width=0.5, edgecolor='white')
ax.set_title('FPS (높을수록 좋음)', fontweight='bold')
ax.set_ylabel('FPS')
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, fps_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'{val:.1f}', ha='center', fontsize=11, fontweight='bold')

# 2) 평균 지연시간 막대
ax = axes[1]
ms_vals = [results_summary[n]['avg_ms'] for n in names]
bars = ax.bar(names, ms_vals, color=colors, width=0.5, edgecolor='white')
ax.set_title('평균 추론 시간 (낮을수록 좋음)', fontweight='bold')
ax.set_ylabel('ms')
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, ms_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1f}ms', ha='center', fontsize=11, fontweight='bold')

# 3) 분포 박스플롯
ax = axes[2]
data = [results_summary[n]['times'] for n in names]
bp = ax.boxplot(data, labels=names, patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_title('추론 시간 분포', fontweight='bold')
ax.set_ylabel('ms')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('runs/compare_speed.png', dpi=150, bbox_inches='tight')
print("\n그래프 저장: runs/compare_speed.png")
plt.show()
