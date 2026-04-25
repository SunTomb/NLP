#!/bin/bash
# =============================================================================
# PopQA always_retrieve 模式评测 — 2 模型 (our + official)
# 用法: CUDA_VISIBLE_DEVICES=1 bash scripts/run_eval_popqa_retrieve.sh
# 预计耗时: ~30-50 分钟 | 显存: ~35-40 GB (单卡)
# =============================================================================

set -eo pipefail

EVAL_SCRIPT="self-rag/retrieval_lm/run_eval_batch_retrieve.py"
RESULT_DIR="results"
LOG_DIR="logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/eval_popqa_retrieve_${TIMESTAMP}.log"

INPUT_FILE="data/eval/popqa_longtail_w_gs.jsonl"
MAX_TOKENS=100
NDOCS=5

declare -A MODELS
MODELS["our"]="outputs/generator_llama2_7b"
MODELS["official"]="models/selfrag_llama2_7b"

mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

echo "============================================================" | tee -a "$LOG_FILE"
echo " PopQA always_retrieve 评测 — $(date)" | tee -a "$LOG_FILE"
echo " 2 models × 1 task (ndocs=${NDOCS})" | tee -a "$LOG_FILE"
echo " Log: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

START_TIME=$(date +%s)
COUNT=0

for model_key in our official; do
    model_path="${MODELS[$model_key]}"
    output_file="${RESULT_DIR}/popqa_retrieve_${model_key}.json"
    COUNT=$((COUNT + 1))

    echo "" | tee -a "$LOG_FILE"
    echo "--- [${COUNT}/2] popqa × ${model_key} (always_retrieve, ndocs=${NDOCS}) ---" | tee -a "$LOG_FILE"
    echo "  模型: ${model_path}" | tee -a "$LOG_FILE"
    echo "  输出: ${output_file}" | tee -a "$LOG_FILE"
    echo "  开始时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"

    TASK_START=$(date +%s)

    python "${EVAL_SCRIPT}" \
        --model_name "${model_path}" \
        --input_file "${INPUT_FILE}" \
        --output_file "${output_file}" \
        --max_new_tokens "${MAX_TOKENS}" \
        --ndocs "${NDOCS}" \
        --metric match \
        --use_groundness \
        --use_utility \
        2>&1 | tee -a "$LOG_FILE"

    TASK_END=$(date +%s)
    TASK_ELAPSED=$(( TASK_END - TASK_START ))
    echo "  ✅ 完成! 耗时: $((TASK_ELAPSED / 60))m $((TASK_ELAPSED % 60))s" | tee -a "$LOG_FILE"
done

# --- 汇总 ---
END_TIME=$(date +%s)
TOTAL_ELAPSED=$(( END_TIME - START_TIME ))

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo " 评测完成! 总耗时: $((TOTAL_ELAPSED / 60))m" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "--- PopQA 结果对比 (no_retrieval vs always_retrieve) ---" | tee -a "$LOG_FILE"
printf "%-20s %-15s %-15s\n" "Model" "no_retrieval" "always_retrieve" | tee -a "$LOG_FILE"
printf "%-20s %-15s %-15s\n" "──────────────" "──────────────" "──────────────" | tee -a "$LOG_FILE"

for model_key in our official; do
    nr_file="${RESULT_DIR}/popqa_${model_key}.json"
    ar_file="${RESULT_DIR}/popqa_retrieve_${model_key}.json"
    nr_score=$(python -c "import json;d=json.load(open('${nr_file}'));print(f\"{d['metric_mean']*100:.2f}%\")" 2>/dev/null || echo "N/A")
    ar_score=$(python -c "import json;d=json.load(open('${ar_file}'));print(f\"{d['metric_mean']*100:.2f}%\")" 2>/dev/null || echo "N/A")
    printf "%-20s %-15s %-15s\n" "$model_key" "$nr_score" "$ar_score" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "详细结果: ${RESULT_DIR}/popqa_retrieve_*.json" | tee -a "$LOG_FILE"
