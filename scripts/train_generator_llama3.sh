#!/bin/bash
# ============================================================
# 改进实验 I1: Llama 3 基座升级 — Generator 训练
# 在集群上执行:
#   conda activate selfrag
#   bash scripts/train_generator_llama3.sh
#
# 硬件需求: 4-8×A100 80GB (DeepSpeed ZeRO-3)
# 预计时间: 1-2 天
# ============================================================
set -e

export PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

MODEL_NAME="models/Llama-3.1-8B"
TRAIN_FILE="data/generator/output_selfrag_training_data.jsonl"
OUTPUT_DIR="outputs/generator_llama3_8b"
NUM_GPUS=8
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE / $NUM_GPUS / $BATCH_SIZE_PER_GPU))

echo "==========================================="
echo " Llama 3 Generator 训练 (改进实验 I1)"
echo "==========================================="
echo " 基座模型: $MODEL_NAME"
echo " 输出目录: $OUTPUT_DIR"
echo " 梯度累积: $GRADIENT_ACC_STEPS"
echo "==========================================="

[ ! -d "$MODEL_NAME" ] && echo "[ERROR] 模型不存在" && exit 1

# 自动查找训练文件
if [ ! -f "$TRAIN_FILE" ]; then
    FOUND=$(ls data/generator/*.jsonl 2>/dev/null | head -1)
    [ -n "$FOUND" ] && TRAIN_FILE="$FOUND" || { echo "[ERROR] 训练数据不存在"; exit 1; }
fi

mkdir -p "$OUTPUT_DIR" logs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

cd self-rag/retrieval_lm

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file stage3_no_offloading_accelerate.conf \
    finetune.py \
    --model_name_or_path "$PROJECT_DIR/$MODEL_NAME" \
    --tokenizer_name "$PROJECT_DIR/$MODEL_NAME" \
    --use_slow_tokenizer \
    --train_file "$PROJECT_DIR/$TRAIN_FILE" \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 3 \
    --output_dir "$PROJECT_DIR/$OUTPUT_DIR" \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1 \
    --use_special_tokens \
    --checkpointing_steps epoch \
    2>&1 | tee "$PROJECT_DIR/logs/generator_llama3_train.log"

cd "$PROJECT_DIR"
echo "[OK] Llama 3 Generator 训练完成！模型保存在: $OUTPUT_DIR"
