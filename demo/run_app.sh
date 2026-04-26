#!/bin/bash
# ============================================================
# I4: 运行交互式 Demo
# ============================================================
set -eo pipefail

cd "$(dirname "$0")/.."
source activate.sh 2>/dev/null || true

PROJECT_DIR=$(pwd)
LOG_FILE="logs/demo_app_$(date +%Y%m%d_%H%M%S).log"

echo "============================================================" | tee -a "$LOG_FILE"
echo " I4: 启动 Self-RAG Demo WebUI — $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

cd demo
python3 app.py --model_path "$PROJECT_DIR/outputs/generator_llama2_7b" 2>&1 | tee -a "$PROJECT_DIR/$LOG_FILE"
