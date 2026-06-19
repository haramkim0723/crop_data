import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

MODELS = {
    'yolo11m': 'runs/yolo11m/results.csv',
    'yolo12m': 'runs/yolo12m/results.csv',
    'yolo26m': 'runs/yolo26m/results.csv',
}
COLORS = {'yolo11m': '#4C72B0', 'yolo12m': '#DD8452', 'yolo26m': '#55A868'}

# ── 1. CSV 로드 ──────────────────────────────────────────────────────────────
dfs = {}
for name, path in MODELS.items():
    if not os.path.exists(path):
        print(f"[경고] {path} 없음 — {name} 제외")
        continue
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    dfs[name] = df

if not dfs:
    print("결과 CSV 파일이 없습니다.")
    exit()

# ── 2. 최고 성능 요약 표 ─────────────────────────────────────────────────────
metrics = [
    ('metrics/precision(B)',   'Precision'),
    ('metrics/recall(B)',      'Recall'),
    ('metrics/mAP50(B)',       'mAP50'),
    ('metrics/mAP50-95(B)',    'mAP50-95'),
]

rows = []
for name, df in dfs.items():
    row = {'모델': name}
    for col, label in metrics:
        val = df[col].max() if col in df.columns else float('nan')
        row[label] = round(val, 4)
    best_epoch = int(df['metrics/mAP50(B)'].idxmax()) + 1
    row['Best Epoch'] = best_epoch
    rows.append(row)

summary = pd.DataFrame(rows).set_index('모델')
print("\n===== 모델 성능 비교 =====")
print(summary.to_string())

# 최고값 강조 출력
print("\n[최고 성능 모델]")
for col in ['Precision', 'Recall', 'mAP50', 'mAP50-95']:
    best = summary[col].idxmax()
    print(f"  {col}: {best} ({summary.loc[best, col]})")

# ── 3. 그래프 ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.suptitle('모델 비교: yolo11m vs yolo12m vs yolo26m', fontsize=16, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

plot_configs = [
    ('train/box_loss',       'Train Box Loss',      0, 0),
    ('train/cls_loss',       'Train Cls Loss',      0, 1),
    ('train/dfl_loss',       'Train DFL Loss',      0, 2),
    ('val/box_loss',         'Val Box Loss',        1, 0),
    ('val/cls_loss',         'Val Cls Loss',        1, 1),
    ('val/dfl_loss',         'Val DFL Loss',        1, 2),
    ('metrics/mAP50(B)',     'mAP50',               2, 0),
    ('metrics/mAP50-95(B)',  'mAP50-95',            2, 1),
    ('metrics/precision(B)', 'Precision & Recall',  2, 2),
]

for col, title, row, col_idx in plot_configs:
    ax = fig.add_subplot(gs[row, col_idx])

    for name, df in dfs.items():
        c = COLORS[name]
        epochs = df['epoch']

        if title == 'Precision & Recall':
            ax.plot(epochs, df['metrics/precision(B)'], color=c, linestyle='-',  label=f'{name} P', linewidth=1.5)
            ax.plot(epochs, df['metrics/recall(B)'],    color=c, linestyle='--', label=f'{name} R', linewidth=1.5)
        elif col in df.columns:
            ax.plot(epochs, df[col], color=c, label=name, linewidth=1.8)

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

# ── 4. 성능 비교 막대 그래프 (별도 figure) ───────────────────────────────────
fig2, axes = plt.subplots(1, 4, figsize=(14, 5))
fig2.suptitle('최고 성능 비교', fontsize=14, fontweight='bold')

bar_metrics = ['Precision', 'Recall', 'mAP50', 'mAP50-95']
model_names = list(summary.index)
x = np.arange(len(model_names))
bar_colors = [COLORS[m] for m in model_names]

for ax, metric in zip(axes, bar_metrics):
    vals = [summary.loc[m, metric] for m in model_names]
    bars = ax.bar(model_names, vals, color=bar_colors, width=0.5, edgecolor='white')
    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.set_ylim(min(vals) * 0.95, min(max(vals) * 1.05, 1.0))
    ax.set_ylabel('Score')
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()

# ── 5. 저장 ──────────────────────────────────────────────────────────────────
fig.savefig('runs/compare_curves.png',  dpi=150, bbox_inches='tight')
fig2.savefig('runs/compare_metrics.png', dpi=150, bbox_inches='tight')
print("\n그래프 저장 완료:")
print("  runs/compare_curves.png  (학습 곡선)")
print("  runs/compare_metrics.png (성능 막대 그래프)")

plt.show()
