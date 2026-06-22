#!/bin/bash
# MLflow 서버를 백그라운드에서 띄우고 학습 실행
# 사용법: bash run_with_mlflow.sh train_yolo12m_v3.py

SCRIPT=${1:-train_yolo12m_v3.py}
MLFLOW_PORT=5000
MLFLOW_DIR="$(pwd)/mlruns"

echo "MLflow 서버 시작 중... (http://127.0.0.1:${MLFLOW_PORT})"
mlflow server \
    --host 127.0.0.1 \
    --port ${MLFLOW_PORT} \
    --backend-store-uri "file://${MLFLOW_DIR}" \
    &

MLFLOW_PID=$!
sleep 2

export MLFLOW_TRACKING_URI="http://127.0.0.1:${MLFLOW_PORT}"
echo "MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}"
echo "학습 시작: ${SCRIPT}"
echo ""

python "${SCRIPT}"

echo ""
echo "학습 완료. MLflow UI: http://127.0.0.1:${MLFLOW_PORT}"
echo "서버를 종료하려면: kill ${MLFLOW_PID}"
