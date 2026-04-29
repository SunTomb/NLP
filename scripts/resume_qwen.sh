#!/bin/bash
# ============================================================
# I1-resume: 从 epoch_1 checkpoint 恢复 Qwen2.5-7B 训练
#
# 原训练在 Tang-3 GPU 4 上因 NAS 磁盘满 (No space left on device)
# 而在 Step 21,635/27,306 (79%) 处中断。
# 此脚本在 Tang-2 的空闲 GPU 上从 epoch_1 恢复最后一个 epoch。
# ============================================================
set -eo pipefail

cd "$(dirname "$0")/.."
source activate.sh 2>/dev/null || true

PROJECT_DIR=$(pwd)
MODEL_DIR="models/Qwen2.5-7B"
OUTPUT_DIR="outputs/generator_qwen2.5_7b"
TRAIN_FILE="data/generator/train.jsonl"
LOG_FILE="logs/train_qwen_resume_$(date +%Y%m%d_%H%M%S).log"
RESUME_CKPT="$PROJECT_DIR/$OUTPUT_DIR/epoch_1"

echo "============================================================" | tee -a "$LOG_FILE"
echo " I1-resume: 恢复 Qwen2.5-7B 训练 — $(date)" | tee -a "$LOG_FILE"
echo " 从 checkpoint: $RESUME_CKPT" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# 验证 checkpoint 完整性
for f in model.safetensors optimizer.bin scheduler.bin random_states_0.pkl; do
    if [ ! -f "$RESUME_CKPT/$f" ]; then
        echo "ERROR: 缺少 $RESUME_CKPT/$f" | tee -a "$LOG_FILE"
        exit 1
    fi
done
echo "✅ Checkpoint 文件完整" | tee -a "$LOG_FILE"

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
    --resume_from_checkpoint "$RESUME_CKPT" \
    2>&1 | tee -a "$PROJECT_DIR/$LOG_FILE"

cd "$PROJECT_DIR"
echo "============================================================" | tee -a "$LOG_FILE"
echo " I1-resume 训练完成 — $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
