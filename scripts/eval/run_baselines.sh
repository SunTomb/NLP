#!/bin/bash
# ============================================================
# Self-RAG 基线模型评测脚本
# 评测 Vanilla Llama 2 和 Standard RAG 基线
#
# 在集群上执行:
#   conda activate selfrag
#   bash scripts/eval/run_baselines.sh
#
# 硬件需求: 1-2×A40 48GB
# ============================================================
set -e

export PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR/self-rag/retrieval_lm"

BASELINE_MODEL="$PROJECT_DIR/models/Llama-2-7b-hf"
EVAL_DATA_DIR="$PROJECT_DIR/data/eval"
OUTPUT_DIR="$PROJECT_DIR/results"

echo "==========================================="
echo " 基线模型评测"
echo "==========================================="

# ==== 1. Vanilla Llama 2 (无检索) ====
echo ""
echo "========== Vanilla Llama 2 (No Retrieval) =========="
VANILLA_DIR="$OUTPUT_DIR/eval_vanilla_llama2"
mkdir -p "$VANILLA_DIR"

# PopQA
echo "[1] Vanilla Llama 2 - PopQA..."
[ -f "$EVAL_DATA_DIR/popqa_longtail_w_gs.jsonl" ] && \
python run_baseline_lm.py \
    --model_name "$BASELINE_MODEL" \
    --input_file "$EVAL_DATA_DIR/popqa_longtail_w_gs.jsonl" \
    --max_new_tokens 100 \
    --metric match \
    --result_fp "$VANILLA_DIR/popqa_results.json" \
    --task qa \
    --prompt_name "prompt_no_input" && echo "  [OK]" || echo "  [SKIP]"

# PubHealth
echo "[2] Vanilla Llama 2 - PubHealth..."
[ -f "$EVAL_DATA_DIR/health_claims_processed.jsonl" ] && \
python run_baseline_lm.py \
    --model_name "$BASELINE_MODEL" \
    --input_file "$EVAL_DATA_DIR/health_claims_processed.jsonl" \
    --max_new_tokens 20 \
    --metric accuracy \
    --result_fp "$VANILLA_DIR/pubhealth_results.json" \
    --task fever && echo "  [OK]" || echo "  [SKIP]"

# ARC-Challenge
echo "[3] Vanilla Llama 2 - ARC-Challenge..."
[ -f "$EVAL_DATA_DIR/arc_challenge_processed.jsonl" ] && \
python run_baseline_lm.py \
    --model_name "$BASELINE_MODEL" \
    --input_file "$EVAL_DATA_DIR/arc_challenge_processed.jsonl" \
    --max_new_tokens 50 \
    --metric match \
    --result_fp "$VANILLA_DIR/arc_results.json" \
    --task arc_c \
    --prompt_name "prompt_no_input" && echo "  [OK]" || echo "  [SKIP]"

# TriviaQA
echo "[4] Vanilla Llama 2 - TriviaQA..."
[ -f "$EVAL_DATA_DIR/triviaqa_test_w_gs.jsonl" ] && \
python run_baseline_lm.py \
    --model_name "$BASELINE_MODEL" \
    --input_file "$EVAL_DATA_DIR/triviaqa_test_w_gs.jsonl" \
    --max_new_tokens 100 \
    --metric match \
    --result_fp "$VANILLA_DIR/triviaqa_results.json" \
    --task qa \
    --prompt_name "prompt_no_input" && echo "  [OK]" || echo "  [SKIP]"

# ==== 2. Standard RAG (始终检索) ====
echo ""
echo "========== Standard RAG (Always Retrieve) =========="
RAG_DIR="$OUTPUT_DIR/eval_standard_rag"
mkdir -p "$RAG_DIR"

# PopQA + RAG
echo "[5] Standard RAG - PopQA..."
[ -f "$EVAL_DATA_DIR/popqa_longtail_w_gs.jsonl" ] && \
python run_baseline_lm.py \
    --model_name "$BASELINE_MODEL" \
    --input_file "$EVAL_DATA_DIR/popqa_longtail_w_gs.jsonl" \
    --max_new_tokens 100 \
    --metric match \
    --result_fp "$RAG_DIR/popqa_results.json" \
    --task qa \
    --mode retrieval \
    --prompt_name "prompt_no_input_retrieval" && echo "  [OK]" || echo "  [SKIP]"

# PubHealth + RAG
echo "[6] Standard RAG - PubHealth..."
[ -f "$EVAL_DATA_DIR/health_claims_processed.jsonl" ] && \
python run_baseline_lm.py \
    --model_name "$BASELINE_MODEL" \
    --input_file "$EVAL_DATA_DIR/health_claims_processed.jsonl" \
    --max_new_tokens 20 \
    --metric accuracy \
    --result_fp "$RAG_DIR/pubhealth_results.json" \
    --task fever \
    --mode retrieval \
    --prompt_name "prompt_no_input_retrieval" && echo "  [OK]" || echo "  [SKIP]"

# ARC-Challenge + RAG
echo "[7] Standard RAG - ARC-Challenge..."
[ -f "$EVAL_DATA_DIR/arc_challenge_processed.jsonl" ] && \
python run_baseline_lm.py \
    --model_name "$BASELINE_MODEL" \
    --input_file "$EVAL_DATA_DIR/arc_challenge_processed.jsonl" \
    --max_new_tokens 50 \
    --metric match \
    --result_fp "$RAG_DIR/arc_results.json" \
    --task arc_c \
    --mode retrieval \
    --prompt_name "prompt_no_input_retrieval" && echo "  [OK]" || echo "  [SKIP]"

# TriviaQA + RAG
echo "[8] Standard RAG - TriviaQA..."
[ -f "$EVAL_DATA_DIR/triviaqa_test_w_gs.jsonl" ] && \
python run_baseline_lm.py \
    --model_name "$BASELINE_MODEL" \
    --input_file "$EVAL_DATA_DIR/triviaqa_test_w_gs.jsonl" \
    --max_new_tokens 100 \
    --metric match \
    --result_fp "$RAG_DIR/triviaqa_results.json" \
    --task qa \
    --mode retrieval \
    --prompt_name "prompt_no_input_retrieval" && echo "  [OK]" || echo "  [SKIP]"

echo ""
echo "==========================================="
echo " 基线评测完成！"
echo "==========================================="
echo " Vanilla Llama 2 结果: $VANILLA_DIR/"
echo " Standard RAG 结果:    $RAG_DIR/"
