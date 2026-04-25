#!/bin/bash
# =============================================================================
# PopQA 完整对比评测 (no_retrieval + always_retrieve)
# 使用包含 ctxs 的 1,399 条数据，确保同一数据集上的公平对比
#
# 用法: CUDA_VISIBLE_DEVICES=1 bash scripts/run_eval_popqa_full.sh
# 预计耗时: ~15-20 分钟 | 显存: ~35-40 GB (单卡)
# =============================================================================

set -eo pipefail

cd "$(dirname "$0")/.."
source activate.sh 2>/dev/null || true

EVAL_BATCH="self-rag/retrieval_lm/run_eval_batch.py"
EVAL_RETRIEVE="self-rag/retrieval_lm/run_eval_batch_retrieve.py"
RESULT_DIR="results"
LOG_DIR="logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/eval_popqa_full_${TIMESTAMP}.log"

# 使用包含 ctxs 的数据（1,399 条）
INPUT_FILE="data/eval/popqa_longtail_w_gs.jsonl"
MAX_TOKENS=100
NDOCS=5

declare -A MODELS
MODELS["our"]="outputs/generator_llama2_7b"
MODELS["official"]="models/selfrag_llama2_7b"

mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

# ---- 验证数据 ----
echo "============================================================" | tee -a "$LOG_FILE"
echo " PopQA 完整对比评测 — $(date)" | tee -a "$LOG_FILE"
echo " 数据: ${INPUT_FILE}" | tee -a "$LOG_FILE"
echo " Log: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

python -c "
import json
count = 0
has_ctxs = 0
with open('${INPUT_FILE}', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        count += 1
        if 'ctxs' in item and len(item['ctxs']) > 0:
            has_ctxs += 1
print(f'  样本数: {count}')
print(f'  含 ctxs: {has_ctxs}/{count}')
" | tee -a "$LOG_FILE"

START_TIME=$(date +%s)
COUNT=0
TOTAL=4  # 2 models × 2 modes

# ==================================================================
# Part 1: no_retrieval 评测 (用同一 1,399 条数据重新评测)
# ==================================================================
echo "" | tee -a "$LOG_FILE"
echo "========== Part 1: no_retrieval 评测 ==========" | tee -a "$LOG_FILE"

for model_key in our official; do
    model_path="${MODELS[$model_key]}"
    output_file="${RESULT_DIR}/popqa_nr_${model_key}.json"
    COUNT=$((COUNT + 1))

    echo "" | tee -a "$LOG_FILE"
    echo "--- [${COUNT}/${TOTAL}] popqa × ${model_key} (no_retrieval) ---" | tee -a "$LOG_FILE"
    echo "  模型: ${model_path}" | tee -a "$LOG_FILE"
    echo "  输出: ${output_file}" | tee -a "$LOG_FILE"
    echo "  开始: $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"

    TASK_START=$(date +%s)

    python "${EVAL_BATCH}" \
        --model_name "${model_path}" \
        --input_file "${INPUT_FILE}" \
        --output_file "${output_file}" \
        --max_new_tokens "${MAX_TOKENS}" \
        --metric match \
        2>&1 | tee -a "$LOG_FILE"

    TASK_END=$(date +%s)
    echo "  ✅ 完成 ($((TASK_END - TASK_START))s)" | tee -a "$LOG_FILE"
done

# ==================================================================
# Part 2: always_retrieve 评测
# ==================================================================
echo "" | tee -a "$LOG_FILE"
echo "========== Part 2: always_retrieve 评测 ==========" | tee -a "$LOG_FILE"

for model_key in our official; do
    model_path="${MODELS[$model_key]}"
    output_file="${RESULT_DIR}/popqa_ar_${model_key}.json"
    COUNT=$((COUNT + 1))

    echo "" | tee -a "$LOG_FILE"
    echo "--- [${COUNT}/${TOTAL}] popqa × ${model_key} (always_retrieve, ndocs=${NDOCS}) ---" | tee -a "$LOG_FILE"
    echo "  模型: ${model_path}" | tee -a "$LOG_FILE"
    echo "  输出: ${output_file}" | tee -a "$LOG_FILE"
    echo "  开始: $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"

    TASK_START=$(date +%s)

    python "${EVAL_RETRIEVE}" \
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
    echo "  ✅ 完成 ($((TASK_END - TASK_START))s)" | tee -a "$LOG_FILE"
done

# ==================================================================
# 汇总对比
# ==================================================================
END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo " 全部完成! 总耗时: $((TOTAL_ELAPSED / 60))m $((TOTAL_ELAPSED % 60))s" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "--- PopQA 结果对比 (1,399 条, 同一数据集) ---" | tee -a "$LOG_FILE"
printf "%-12s %-15s %-15s\n" "Model" "no_retrieval" "always_retrieve" | tee -a "$LOG_FILE"
printf "%-12s %-15s %-15s\n" "──────────" "──────────────" "──────────────" | tee -a "$LOG_FILE"

for model_key in our official; do
    nr_file="${RESULT_DIR}/popqa_nr_${model_key}.json"
    ar_file="${RESULT_DIR}/popqa_ar_${model_key}.json"
    nr_score=$(python -c "import json;d=json.load(open('${nr_file}'));print(f\"{d['metric_mean']*100:.2f}%\")" 2>/dev/null || echo "N/A")
    ar_score=$(python -c "import json;d=json.load(open('${ar_file}'));print(f\"{d['metric_mean']*100:.2f}%\")" 2>/dev/null || echo "N/A")
    printf "%-12s %-15s %-15s\n" "$model_key" "$nr_score" "$ar_score" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "论文参考: no_ret=32.7%, always_ret=54.9%" | tee -a "$LOG_FILE"
echo "结果文件: ${RESULT_DIR}/popqa_{nr,ar}_{our,official}.json" | tee -a "$LOG_FILE"
