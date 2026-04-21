#!/bin/bash
# ============================================================
# Self-RAG 全面评测脚本
# 在集群上执行:
#   conda activate selfrag
#   bash scripts/eval/run_all_eval.sh
#
# 硬件需求: 1-2×A100 或 A40 (推理)
# 预计时间: 每个评测任务 1-4 小时
# ============================================================
set -e

export PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR/self-rag/retrieval_lm"

# ---- 配置 ----
# 我们复现的模型
OUR_MODEL="$PROJECT_DIR/outputs/generator_llama2_7b"
# 官方预训练模型（交叉验证）
OFFICIAL_MODEL="$PROJECT_DIR/models/selfrag_llama2_7b"
# Vanilla Llama 2 基线
BASELINE_MODEL="$PROJECT_DIR/models/Llama-2-7b-hf"

# 选择要评测的模型
MODEL="${1:-$OFFICIAL_MODEL}"  # 默认先用官方模型验证
MODEL_TAG="${2:-official_7b}"

EVAL_DATA_DIR="$PROJECT_DIR/data/eval"
OUTPUT_DIR="$PROJECT_DIR/results/eval_${MODEL_TAG}"
mkdir -p "$OUTPUT_DIR"

echo "==========================================="
echo " Self-RAG 评测"
echo "==========================================="
echo " 模型:    $MODEL"
echo " 标签:    $MODEL_TAG"
echo " 数据:    $EVAL_DATA_DIR"
echo " 输出:    $OUTPUT_DIR"
echo "==========================================="

# ==== Short-form 评测 ====

# ---- 1. PopQA ----
echo ""
echo "[1/6] PopQA (EM) ..."
if [ -f "$EVAL_DATA_DIR/popqa_longtail_w_gs.jsonl" ]; then
    python run_short_form.py \
        --model_name "$MODEL" \
        --input_file "$EVAL_DATA_DIR/popqa_longtail_w_gs.jsonl" \
        --mode adaptive_retrieval \
        --max_new_tokens 100 \
        --threshold 0.2 \
        --output_file "$OUTPUT_DIR/popqa_results.json" \
        --metric match \
        --ndocs 10 \
        --use_groundness --use_utility --use_seqscore \
        --dtype half
    echo "[OK] PopQA 完成"
else
    echo "[SKIP] PopQA 数据文件不存在"
fi

# ---- 2. TriviaQA ----
echo ""
echo "[2/6] TriviaQA (EM) ..."
if [ -f "$EVAL_DATA_DIR/triviaqa_test_w_gs.jsonl" ]; then
    python run_short_form.py \
        --model_name "$MODEL" \
        --input_file "$EVAL_DATA_DIR/triviaqa_test_w_gs.jsonl" \
        --mode adaptive_retrieval \
        --max_new_tokens 100 \
        --threshold 0.2 \
        --output_file "$OUTPUT_DIR/triviaqa_results.json" \
        --metric match \
        --ndocs 10 \
        --use_groundness --use_utility --use_seqscore \
        --dtype half
    echo "[OK] TriviaQA 完成"
else
    echo "[SKIP] TriviaQA 数据文件不存在"
fi

# ---- 3. PubHealth ----
echo ""
echo "[3/6] PubHealth (Accuracy) ..."
if [ -f "$EVAL_DATA_DIR/health_claims_processed.jsonl" ]; then
    python run_short_form.py \
        --model_name "$MODEL" \
        --input_file "$EVAL_DATA_DIR/health_claims_processed.jsonl" \
        --mode adaptive_retrieval \
        --max_new_tokens 50 \
        --threshold 0.2 \
        --output_file "$OUTPUT_DIR/pubhealth_results.json" \
        --metric match \
        --ndocs 5 \
        --use_groundness --use_utility --use_seqscore \
        --task fever \
        --dtype half
    echo "[OK] PubHealth 完成"
else
    echo "[SKIP] PubHealth 数据文件不存在"
fi

# ---- 4. ARC-Challenge ----
echo ""
echo "[4/6] ARC-Challenge (Accuracy) ..."
if [ -f "$EVAL_DATA_DIR/arc_challenge_processed.jsonl" ]; then
    python run_short_form.py \
        --model_name "$MODEL" \
        --input_file "$EVAL_DATA_DIR/arc_challenge_processed.jsonl" \
        --mode adaptive_retrieval \
        --max_new_tokens 50 \
        --threshold 0.2 \
        --output_file "$OUTPUT_DIR/arc_results.json" \
        --metric match \
        --ndocs 5 \
        --use_groundness --use_utility --use_seqscore \
        --task arc_c \
        --dtype half
    echo "[OK] ARC-Challenge 完成"
else
    echo "[SKIP] ARC-Challenge 数据文件不存在"
fi

# ==== Long-form 评测 ====

# ---- 5. ASQA ----
echo ""
echo "[5/6] ASQA (Correctness + Citation P/R) ..."
if [ -f "$EVAL_DATA_DIR/asqa_eval_gtr_top100.json" ]; then
    python run_long_form_static.py \
        --model_name "$MODEL" \
        --ndocs 5 \
        --max_new_tokens 300 \
        --threshold 0.2 \
        --use_grounding --use_utility --use_seqscore \
        --task asqa \
        --input_file "$EVAL_DATA_DIR/asqa_eval_gtr_top100.json" \
        --output_file "$OUTPUT_DIR/asqa_results.json" \
        --max_depth 7 \
        --mode always_retrieve \
        --dtype half
    echo "[OK] ASQA 完成"
else
    echo "[SKIP] ASQA 数据文件不存在"
fi

# ---- 6. FactScore ----
echo ""
echo "[6/6] FactScore ..."
if [ -f "$EVAL_DATA_DIR/factscore_unlabeled_alpaca_13b_retrieval.jsonl" ]; then
    python run_long_form_static.py \
        --model_name "$MODEL" \
        --ndocs 5 \
        --max_new_tokens 300 \
        --threshold 0.2 \
        --use_grounding --use_utility --use_seqscore \
        --task factscore \
        --input_file "$EVAL_DATA_DIR/factscore_unlabeled_alpaca_13b_retrieval.jsonl" \
        --output_file "$OUTPUT_DIR/factscore_results.json" \
        --max_depth 7 \
        --dtype half
    echo "[OK] FactScore 完成"
else
    echo "[SKIP] FactScore 数据文件不存在"
fi

echo ""
echo "==========================================="
echo " 评测完成！"
echo "==========================================="
echo " 结果保存在: $OUTPUT_DIR/"
ls -la "$OUTPUT_DIR/"
echo ""
echo "汇总结果: python $PROJECT_DIR/scripts/eval/summarize_results.py --result_dir $OUTPUT_DIR"
