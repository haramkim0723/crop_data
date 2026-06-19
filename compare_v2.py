import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

MODELS = {
    'yolo11m (v1)': 'runs/yolo11m/results.csv',
    'yolo11m (v2)': 'runs/yolo11m_v2/results.csv',
    'yolo12m (v1)': 'runs/yolo12m/results.csv',
    'yolo12m (v2)': 'runs/yolo12m_v2/results.csv',
}
COLORS = {
    'yolo11m (v1)': '#4C72B0',
    'yolo11m (v2)': '#A8C4E0',
    'yolo12m (v1)': '#DD8452',
    'yolo12m (v2)': '#F5C199',
}

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
    ('metrics/precision(B)',  'Precision'),
    ('metrics/recall(B)',     'Recall'),
    ('metrics/mAP50(B)',      'mAP50'),
    ('metrics/mAP50-95(B)',   'mAP50-95'),
]

rows = []
for name, df in dfs.items():
    row = {'모델': name}
    for col, label in metrics:
        val = df[col].max() if col in df.columns else float('nan')
        row[label] = round(val, 4)
    row['Best Epoch'] = int(df['metrics/mAP50(B)'].idxmax()) + 1
    row['Total Epochs'] = len(df)
    rows.append(row)

summary = pd.DataFrame(rows).set_index('모델')
print("\n===== v1 vs v2 성능 비교 =====")
print(summary.to_string())

# v1 → v2 개선량
print("\n[v2 개선량]")
for model in ['yolo11m', 'yolo12m']:
    v1 = f'{model} (v1)'
    v2 = f'{model} (v2)'
    if v1 in summary.index and v2 in summary.index:
        print(f"\n  {model}:")
        for col in ['Precision', 'Recall', 'mAP50', 'mAP50-95']:
            diff = summary.loc[v2, col] - summary.loc[v1, col]
            sign = '+' if diff >= 0 else ''
            print(f"    {col}: {sign}{diff:.4f}")

# ── 3. 학습 곡선 비교 ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 10))
fig.suptitle('v1 vs v2 학습 곡선 비교', fontsize=15, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

plot_configs = [
    ('metrics/mAP50(B)',      'mAP50',        0, 0),
    ('metrics/mAP50-95(B)',   'mAP50-95',     0, 1),
    ('metrics/precision(B)',  'Precision',    0, 2),
    ('val/box_loss',          'Val Box Loss', 1, 0),
    ('val/cls_loss',          'Val Cls Loss', 1, 1),
    ('metrics/recall(B)',     'Recall',       1, 2),
]

for col, title, row, col_idx in plot_configs:
    ax = fig.add_subplot(gs[row, col_idx])
    for name, df in dfs.items():
        if col in df.columns:
            style = '-' if 'v1' in name else '--'
            ax.plot(df['epoch'], df[col], color=COLORS[name],
                    linestyle=style, label=name, linewidth=1.8)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

# ── 4. 성능 비교 막대 그래프 ─────────────────────────────────────────────────
fig2, axes = plt.subplots(1, 4, figsize=(16, 5))
fig2.suptitle('v1 vs v2 최고 성능 비교', fontsize=13, fontweight='bold')

bar_metrics = ['Precision', 'Recall', 'mAP50', 'mAP50-95']
model_names = list(summary.index)
x = np.arange(len(model_names))
bar_colors = [COLORS[m] for m in model_names]

for ax, metric in zip(axes, bar_metrics):
    vals = [summary.loc[m, metric] for m in model_names]
    bars = ax.bar(model_names, vals, color=bar_colors, width=0.5, edgecolor='white')
    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.set_ylim(min(vals) * 0.97, min(max(vals) * 1.03, 1.0))
    ax.set_ylabel('Score')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=15, fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()

# ── 5. 저장 ──────────────────────────────────────────────────────────────────
fig.savefig('runs/compare_v2_curves.png',  dpi=150, bbox_inches='tight')
fig2.savefig('runs/compare_v2_metrics.png', dpi=150, bbox_inches='tight')
print("\n그래프 저장 완료:")
print("  runs/compare_v2_curves.png")
print("  runs/compare_v2_metrics.png")

plt.show()
