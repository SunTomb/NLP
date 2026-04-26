#!/bin/bash
# ============================================================
# I1: Self-RAG Qwen2.5 基座微调实验
#
# 此脚本将在单卡 A40 上使用 Adafactor 优化器和 Gradient Checkpointing
# 对 Qwen2.5-7B 进行训练，复刻 Llama-2 上的实验配置。
# ============================================================
set -eo pipefail

cd "$(dirname "$0")/.."
source activate.sh 2>/dev/null || true

PROJECT_DIR=$(pwd)
MODEL_REPO="Qwen/Qwen2.5-7B"
MODEL_DIR="models/Qwen2.5-7B"
OUTPUT_DIR="outputs/generator_qwen2.5_7b"
TRAIN_FILE="data/generator/train.jsonl"
LOG_FILE="logs/train_qwen_$(date +%Y%m%d_%H%M%S).log"

echo "============================================================" | tee -a "$LOG_FILE"
echo " I1: 启动 Qwen2.5-7B 微调流程 — $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# 1. 确保模型已下载
if [ ! -d "$MODEL_DIR" ]; then
    echo "[1/2] 正在下载 $MODEL_REPO 到 $MODEL_DIR ..." | tee -a "$LOG_FILE"
    python3 -c "
from huggingface_hub import snapshot_download
import os
os.makedirs('$MODEL_DIR', exist_ok=True)
snapshot_download(repo_id='$MODEL_REPO', local_dir='$MODEL_DIR', local_files_only=False)
" 2>&1 | tee -a "$LOG_FILE"
else
    echo "[1/2] 模型 $MODEL_DIR 已存在，跳过下载。" | tee -a "$LOG_FILE"
fi

# 2. 启动训练
echo "[2/2] 开始训练 (单卡 A40, Adafactor, GradCkpt)..." | tee -a "$LOG_FILE"
echo "日志输出: $LOG_FILE" | tee -a "$LOG_FILE"

mkdir -p "$OUTPUT_DIR"

cd self-rag/retrieval_lm

export LD_LIBRARY_PATH=/NAS/yesh/NLP/.conda/selfrag/lib:$LD_LIBRARY_PATH

python3 finetune.py \
    --model_name_or_path "$PROJECT_DIR/$MODEL_DIR" \
    --tokenizer_name "$PROJECT_DIR/$MODEL_DIR" \
    --use_slow_tokenizer \
    --train_file "$PROJECT_DIR/$TRAIN_FILE" \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 3 \
    --output_dir "$PROJECT_DIR/$OUTPUT_DIR" \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1 \
    --use_special_tokens \
    --checkpointing_steps epoch \
    2>&1 | tee -a "$PROJECT_DIR/$LOG_FILE"

cd "$PROJECT_DIR"
echo "============================================================" | tee -a "$LOG_FILE"
echo " I1 训练完成 — $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
