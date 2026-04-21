#!/bin/bash
# ============================================================
# Self-RAG Critic 训练脚本
# 在集群上执行:
#   conda activate selfrag
#   bash scripts/train_critic.sh
#
# 硬件需求: 2×A100 80GB (FSDP)
# 预计时间: 8-12 小时
# 显存使用: ~35GB/卡
# ============================================================
set -e

export PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

# ---- 配置 ----
MODEL_NAME="models/Llama-2-7b-hf"
DATA_PATH="data/critic/critic_train_data.json"
OUTPUT_DIR="outputs/critic_llama2_7b"
NUM_GPUS=2
NUM_EPOCHS=3
LEARNING_RATE=2e-5
BATCH_SIZE=1
GRAD_ACCUM=8
MAX_SEQ_LEN=512
MASTER_PORT=2568

echo "==========================================="
echo " Self-RAG Critic 训练"
echo "==========================================="
echo " 基座模型:      $MODEL_NAME"
echo " 训练数据:      $DATA_PATH"
echo " 输出目录:      $OUTPUT_DIR"
echo " GPU 数量:      $NUM_GPUS"
echo " Epoch:         $NUM_EPOCHS"
echo " 学习率:        $LEARNING_RATE"
echo " Batch/GPU:     $BATCH_SIZE"
echo " 梯度累积:      $GRAD_ACCUM"
echo " 有效 Batch:    $(($NUM_GPUS * $BATCH_SIZE * $GRAD_ACCUM))"
echo "==========================================="

# 检查数据是否存在
if [ ! -f "$DATA_PATH" ]; then
    echo "[ERROR] 训练数据不存在: $DATA_PATH"
    echo "  请先运行: bash scripts/download_data.sh"
    exit 1
fi

# 检查模型是否存在
if [ ! -d "$MODEL_NAME" ]; then
    echo "[ERROR] 基座模型不存在: $MODEL_NAME"
    echo "  请先运行: bash scripts/download_models.sh"
    exit 1
fi

# 创建输出和日志目录
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

echo ""
echo "[INFO] 开始训练..."
echo "[INFO] 日志文件: logs/critic_train.log"
echo "[INFO] 建议使用 tmux/screen 运行，防止 SSH 断连"
echo ""

# ---- 训练命令 ----
cd self-rag/data_creation

torchrun --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    train_special_tokens.py \
    --model_name_or_path "$PROJECT_DIR/$MODEL_NAME" \
    --data_path "$PROJECT_DIR/$DATA_PATH" \
    --use_special_token True \
    --bf16 True \
    --output_dir "$PROJECT_DIR/$OUTPUT_DIR" \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 2 \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0. \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length $MAX_SEQ_LEN \
    --fsdp "full_shard auto_wrap" \
    2>&1 | tee "$PROJECT_DIR/logs/critic_train.log"

cd "$PROJECT_DIR"

echo ""
echo "==========================================="
echo " Critic 训练完成！"
echo "==========================================="
echo " 模型保存在: $OUTPUT_DIR"
echo " 日志文件:   logs/critic_train.log"
echo ""
echo "后续步骤:"
echo "  1. 检查 loss 曲线"
echo "  2. 运行 Critic 准确率评测"
echo "  3. 开始 Generator 训练: bash scripts/train_generator.sh"
