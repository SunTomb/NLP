#!/bin/bash
# ============================================================
# I2: 中文适配微调 (基于 Qwen2.5-7B)
# ============================================================
set -eo pipefail

cd "$(dirname "$0")/.."
source activate.sh 2>/dev/null || true

PROJECT_DIR=$(pwd)
MODEL_DIR="models/Qwen2.5-7B"
OUTPUT_DIR="outputs/generator_qwen2.5_7b_zh"
TRAIN_FILE="data/generator/train_zh_dummy.jsonl"
LOG_FILE="logs/train_zh_$(date +%Y%m%d_%H%M%S).log"

echo "============================================================" | tee -a "$LOG_FILE"
echo " I2: 启动中文适配训练流程 — $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# 1. 准备数据
python3 scripts/prepare_zh_dummy.py 2>&1 | tee -a "$LOG_FILE"

if [ ! -f "$TRAIN_FILE" ]; then
    echo "[ERROR] 数据生成失败，无法找到 $TRAIN_FILE" | tee -a "$LOG_FILE"
    exit 1
fi

# 2. 检查模型
if [ ! -d "$MODEL_DIR" ]; then
    echo "[ERROR] $MODEL_DIR 不存在。请先运行 I1 的 Qwen 下载脚本。" | tee -a "$LOG_FILE"
    exit 1
fi

# 3. 启动训练
echo "[2/2] 开始训练 (单卡 A40, Adafactor, GradCkpt)..." | tee -a "$LOG_FILE"
mkdir -p "$OUTPUT_DIR"
cd self-rag/retrieval_lm

export LD_LIBRARY_PATH=/NAS/yesh/NLP/.conda/selfrag/lib:$LD_LIBRARY_PATH

python3 finetune.py \
    --model_name_or_path "$PROJECT_DIR/$MODEL_DIR" \
    --tokenizer_name "$PROJECT_DIR/$MODEL_DIR" \
    --use_slow_tokenizer \
    --train_file "$PROJECT_DIR/$TRAIN_FILE" \
    --max_seq_length 2048 \
    --preprocessing_num_workers 8 \
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
echo " I2 训练完成 — $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
