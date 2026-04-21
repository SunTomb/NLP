#!/bin/bash
# ============================================================
# Self-RAG Generator 训练脚本
# 在集群上执行:
#   conda activate selfrag
#   bash scripts/train_generator.sh
#
# 硬件需求: 4-8×A100 80GB (DeepSpeed ZeRO-3)
# 预计时间: 1-2 天
# 显存使用: ~25GB/卡 (8 GPU) 或 ~35GB/卡 (4 GPU)
# ============================================================
set -e

export PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

# ---- 配置 ----
MODEL_NAME="models/Llama-2-7b-hf"
TOKENIZER_NAME="models/Llama-2-7b-hf"
TRAIN_FILE="data/generator/output_selfrag_training_data.jsonl"  # 请确认实际文件名
OUTPUT_DIR="outputs/generator_llama2_7b"
MODEL_SIZE="7B"
NUM_GPUS=8          # 使用 8×A100，显存充裕可增大 batch
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE / $NUM_GPUS / $BATCH_SIZE_PER_GPU))
MAX_SEQ_LEN=2048
NUM_EPOCHS=3
LEARNING_RATE=2e-5

echo "==========================================="
echo " Self-RAG Generator 训练 ($MODEL_SIZE)"
echo "==========================================="
echo " 基座模型:      $MODEL_NAME"
echo " 训练数据:      $TRAIN_FILE"
echo " 输出目录:      $OUTPUT_DIR"
echo " GPU 数量:      $NUM_GPUS"
echo " Batch/GPU:     $BATCH_SIZE_PER_GPU"
echo " 有效 Batch:    $TOTAL_BATCH_SIZE"
echo " 梯度累积:      $GRADIENT_ACC_STEPS"
echo " 最大序列长度:  $MAX_SEQ_LEN"
echo " Epoch:         $NUM_EPOCHS"
echo " 学习率:        $LEARNING_RATE"
echo "==========================================="

# 检查训练数据
if [ ! -f "$TRAIN_FILE" ]; then
    # 尝试寻找实际文件
    FOUND_FILE=$(ls data/generator/*.jsonl 2>/dev/null | head -1)
    if [ -n "$FOUND_FILE" ]; then
        echo "[INFO] 使用找到的训练文件: $FOUND_FILE"
        TRAIN_FILE="$FOUND_FILE"
    else
        echo "[ERROR] 训练数据不存在: $TRAIN_FILE"
        echo "  请先运行: bash scripts/download_data.sh"
        echo "  或检查 data/generator/ 目录下的文件名"
        exit 1
    fi
fi

# 检查模型
if [ ! -d "$MODEL_NAME" ]; then
    echo "[ERROR] 基座模型不存在: $MODEL_NAME"
    exit 1
fi

# 创建目录
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

echo ""
echo "[INFO] 开始 Generator 训练..."
echo "[INFO] 日志文件: logs/generator_train.log"
echo "[INFO] 强烈建议使用 tmux/screen 运行！"
echo ""

# ---- DeepSpeed ZeRO-3 配置 ----
cd self-rag/retrieval_lm

# 设置 GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file stage3_no_offloading_accelerate.conf \
    finetune.py \
    --model_name_or_path "$PROJECT_DIR/$MODEL_NAME" \
    --tokenizer_name "$PROJECT_DIR/$TOKENIZER_NAME" \
    --use_slow_tokenizer \
    --train_file "$PROJECT_DIR/$TRAIN_FILE" \
    --max_seq_length $MAX_SEQ_LEN \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs $NUM_EPOCHS \
    --output_dir "$PROJECT_DIR/$OUTPUT_DIR" \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1 \
    --use_special_tokens \
    --checkpointing_steps epoch \
    2>&1 | tee "$PROJECT_DIR/logs/generator_train.log"

cd "$PROJECT_DIR"

echo ""
echo "==========================================="
echo " Generator 训练完成！"
echo "==========================================="
echo " 模型保存在: $OUTPUT_DIR"
echo " 日志文件:   logs/generator_train.log"
echo " TensorBoard: tensorboard --logdir $OUTPUT_DIR"
echo ""
echo "后续步骤:"
echo "  1. 绘制 loss 曲线: python scripts/plot_loss.py"
echo "  2. 快速验证: python scripts/quick_inference.py --model_path $OUTPUT_DIR"
echo "  3. 全面评测:  bash scripts/eval/run_all_eval.sh"
