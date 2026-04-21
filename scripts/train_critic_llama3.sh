#!/bin/bash
# ============================================================
# 改进实验 I1: Llama 3 基座升级 — Critic 训练
# 在集群上执行:
#   conda activate selfrag
#   bash scripts/train_critic_llama3.sh
#
# 硬件需求: 2×A100 80GB (FSDP)
# 预计时间: 8-12 小时
# ============================================================
set -e

export PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

MODEL_NAME="models/Llama-3.1-8B"
DATA_PATH="data/critic/critic_train_data.json"
OUTPUT_DIR="outputs/critic_llama3_8b"
NUM_GPUS=2
MASTER_PORT=2569

echo "==========================================="
echo " Llama 3 Critic 训练 (改进实验 I1)"
echo "==========================================="
echo " 基座模型: $MODEL_NAME"
echo " 输出目录: $OUTPUT_DIR"
echo "==========================================="

[ ! -d "$MODEL_NAME" ] && echo "[ERROR] 模型不存在，请先下载 Llama 3.1 8B" && exit 1
[ ! -f "$DATA_PATH" ] && echo "[ERROR] 训练数据不存在" && exit 1

mkdir -p "$OUTPUT_DIR" logs

cd self-rag/data_creation

torchrun --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    train_special_tokens.py \
    --model_name_or_path "$PROJECT_DIR/$MODEL_NAME" \
    --data_path "$PROJECT_DIR/$DATA_PATH" \
    --use_special_token True \
    --bf16 True \
    --output_dir "$PROJECT_DIR/$OUTPUT_DIR" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 512 \
    --fsdp "full_shard auto_wrap" \
    2>&1 | tee "$PROJECT_DIR/logs/critic_llama3_train.log"

cd "$PROJECT_DIR"
echo "[OK] Llama 3 Critic 训练完成！模型保存在: $OUTPUT_DIR"
