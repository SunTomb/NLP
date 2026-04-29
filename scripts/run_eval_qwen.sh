#!/bin/bash
# =============================================================================
# Qwen2.5-7B Self-RAG 评测脚本
# 评测内容: PopQA (no_ret + always_ret) + ARC-C + TriviaQA
# =============================================================================
set -eo pipefail
cd "$(dirname "$0")/.."
source activate.sh 2>/dev/null || true

EVAL_BATCH="self-rag/retrieval_lm/run_eval_batch.py"
EVAL_RETRIEVE="self-rag/retrieval_lm/run_eval_batch_retrieve.py"
RESULT_DIR="results"
LOG_DIR="logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/eval_qwen_${TIMESTAMP}.log"

QWEN_MODEL="outputs/generator_qwen2.5_7b"

mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

echo "============================================================" | tee -a "$LOG_FILE"
echo " Qwen2.5-7B Self-RAG 评测 — $(date)" | tee -a "$LOG_FILE"
echo " 模型: ${QWEN_MODEL}" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

START_TIME=$(date +%s)

# 1. PopQA no_retrieval
echo "" | tee -a "$LOG_FILE"
echo "--- [1/4] PopQA no_retrieval ---" | tee -a "$LOG_FILE"
python "${EVAL_BATCH}" \
    --model_name "${QWEN_MODEL}" \
    --input_file "data/eval/popqa_longtail_w_gs.jsonl" \
    --output_file "${RESULT_DIR}/popqa_nr_qwen.json" \
    --max_new_tokens 100 \
    --metric match \
    2>&1 | tee -a "$LOG_FILE"
echo "  done $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"

# 2. PopQA always_retrieve
echo "" | tee -a "$LOG_FILE"
echo "--- [2/4] PopQA always_retrieve ---" | tee -a "$LOG_FILE"
python "${EVAL_RETRIEVE}" \
    --model_name "${QWEN_MODEL}" \
    --input_file "data/eval/popqa_longtail_w_gs.jsonl" \
    --output_file "${RESULT_DIR}/popqa_ar_qwen.json" \
    --max_new_tokens 100 \
    --ndocs 5 \
    --metric match \
    --use_groundness \
    --use_utility \
    2>&1 | tee -a "$LOG_FILE"
echo "  done $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"

# 3. ARC-C
echo "" | tee -a "$LOG_FILE"
echo "--- [3/4] ARC-C no_retrieval ---" | tee -a "$LOG_FILE"
python "${EVAL_BATCH}" \
    --model_name "${QWEN_MODEL}" \
    --input_file "data/eval/arc_challenge_processed.jsonl" \
    --output_file "${RESULT_DIR}/arc_qwen.json" \
    --max_new_tokens 50 \
    --metric match \
    --task arc_c \
    2>&1 | tee -a "$LOG_FILE"
echo "  done $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"

# 4. TriviaQA
echo "" | tee -a "$LOG_FILE"
echo "--- [4/4] TriviaQA no_retrieval ---" | tee -a "$LOG_FILE"
python "${EVAL_BATCH}" \
    --model_name "${QWEN_MODEL}" \
    --input_file "data/eval/triviaqa_test_w_gs.jsonl" \
    --output_file "${RESULT_DIR}/triviaqa_qwen.json" \
    --max_new_tokens 100 \
    --metric match \
    2>&1 | tee -a "$LOG_FILE"
echo "  done $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"

# 汇总
END_TIME=$(date +%s)
echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo " 全部完成! 耗时: $(( (END_TIME-START_TIME)/60 ))m" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

for f in popqa_nr_qwen popqa_ar_qwen arc_qwen triviaqa_qwen; do
    rf="${RESULT_DIR}/${f}.json"
    if [ -f "$rf" ]; then
        score=$(python -c "import json;d=json.load(open('${rf}'));print(f\"{d['metric_mean']*100:.2f}%\")" 2>/dev/null || echo "ERR")
        echo "  ${f}: ${score}" | tee -a "$LOG_FILE"
    fi
done
