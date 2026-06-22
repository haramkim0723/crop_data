"""
기존 실험 결과(results.csv + args.yaml)를 MLflow에 업로드.
사용법:
    export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
    python upload_to_mlflow.py
"""
import os
import yaml
import pandas as pd
import mlflow
from pathlib import Path

RUNS_DIR = Path("runs")
EXPERIMENT_NAME = "sugar_beet_yolo"

RUN_DIRS = [
    "yolo11m",
    "yolo11m_v2",
    "yolo12m",
    "yolo12m_v2",
    "yolo26m",
]

# 기록할 메트릭 컬럼
METRIC_COLS = [
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
]

# args.yaml에서 파라미터로 기록할 항목
PARAM_KEYS = [
    "model", "epochs", "imgsz", "batch", "patience",
    "optimizer", "lr0", "lrf", "momentum", "weight_decay",
    "warmup_epochs", "cos_lr", "close_mosaic",
    "box", "cls", "dfl",
    "hsv_h", "hsv_s", "hsv_v",
    "degrees", "translate", "scale", "flipud", "fliplr",
    "mosaic", "mixup", "copy_paste",
    "augment", "dropout", "amp", "seed",
]


def upload_run(run_dir: Path, experiment_id: str):
    results_csv = run_dir / "results.csv"
    args_yaml   = run_dir / "args.yaml"

    if not results_csv.exists():
        print(f"[건너뜀] {run_dir.name}: results.csv 없음")
        return

    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()

    params = {}
    if args_yaml.exists():
        with open(args_yaml) as f:
            args = yaml.safe_load(f)
        params = {k: str(args[k]) for k in PARAM_KEYS if k in args}

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_dir.name):
        # 파라미터 기록
        if params:
            mlflow.log_params(params)

        # epoch별 메트릭 기록 (MLflow는 괄호 불허 → 치환)
        def sanitize(name: str) -> str:
            return name.replace("(B)", "").replace("/", ".").strip(".")

        for _, row in df.iterrows():
            step = int(row["epoch"])
            metrics = {
                sanitize(col): float(row[col])
                for col in METRIC_COLS
                if col in df.columns and pd.notna(row[col])
            }
            mlflow.log_metrics(metrics, step=step)

        # 아티팩트: results.png, confusion_matrix.png
        for artifact in ["results.png", "confusion_matrix_normalized.png", "args.yaml"]:
            p = run_dir / artifact
            if p.exists():
                mlflow.log_artifact(str(p))

        print(f"[완료] {run_dir.name} — {len(df)}epoch 업로드")


def main():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        experiment_id = client.create_experiment(EXPERIMENT_NAME)
        print(f"실험 생성: {EXPERIMENT_NAME} (id={experiment_id})")
    else:
        experiment_id = exp.experiment_id
        print(f"기존 실험 사용: {EXPERIMENT_NAME} (id={experiment_id})")

    for name in RUN_DIRS:
        run_dir = RUNS_DIR / name
        if run_dir.exists():
            upload_run(run_dir, experiment_id)
        else:
            print(f"[건너뜀] {name}: 폴더 없음")

    print(f"\nMLflow UI: {tracking_uri}")


if __name__ == "__main__":
    main()
