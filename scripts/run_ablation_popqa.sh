#!/bin/bash
# =============================================================================
# I3: 消融实验 — 在 PopQA always_retrieve 上测试不同 scoring 组合
#
# 用法: CUDA_VISIBLE_DEVICES=1 bash scripts/run_ablation_popqa.sh
# 预计耗时: ~30-40 分钟 (3 个变体 × ~10 分钟/变体)
# 显存: ~35-40 GB
# =============================================================================

set -eo pipefail

cd "$(dirname "$0")/.."
source activate.sh 2>/dev/null || true

EVAL_RETRIEVE="self-rag/retrieval_lm/run_eval_batch_retrieve.py"
RESULT_DIR="results"
LOG_DIR="logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/ablation_popqa_${TIMESTAMP}.log"

INPUT_FILE="data/eval/popqa_longtail_w_gs.jsonl"
MODEL="outputs/generator_llama2_7b"
MAX_TOKENS=100
NDOCS=5

mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

echo "============================================================" | tee -a "$LOG_FILE"
echo " I3: PopQA 消融实验 — $(date)" | tee -a "$LOG_FILE"
echo " 模型: ${MODEL}" | tee -a "$LOG_FILE"
echo " 数据: ${INPUT_FILE}" | tee -a "$LOG_FILE"
echo " Log: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

START_TIME=$(date +%s)
COUNT=0
TOTAL=3

# ---- 基线: Full model (已有结果 popqa_ar_our.json = 50.46%) ----
echo "" | tee -a "$LOG_FILE"
echo "基线 (Full): results/popqa_ar_our.json" | tee -a "$LOG_FILE"
if [ -f "${RESULT_DIR}/popqa_ar_our.json" ]; then
    BASELINE=$(python -c "import json;d=json.load(open('${RESULT_DIR}/popqa_ar_our.json'));print(f\"{d['metric_mean']*100:.2f}%\")")
    echo "  Score: ${BASELINE} (已有结果，跳过)" | tee -a "$LOG_FILE"
else
    echo "  ⚠️ 未找到基线结果，请先运行 run_eval_popqa_full.sh" | tee -a "$LOG_FILE"
fi

# ---- 变体 1: 去掉 groundness scoring ----
COUNT=$((COUNT + 1))
echo "" | tee -a "$LOG_FILE"
echo "--- [${COUNT}/${TOTAL}] w/o Groundness (只保留 Utility) ---" | tee -a "$LOG_FILE"
echo "  开始: $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"

TASK_START=$(date +%s)
python "${EVAL_RETRIEVE}" \
    --model_name "${MODEL}" \
    --input_file "${INPUT_FILE}" \
    --output_file "${RESULT_DIR}/popqa_ar_our_no_grd.json" \
    --max_new_tokens "${MAX_TOKENS}" \
    --ndocs "${NDOCS}" \
    --metric match \
    --use_utility \
    2>&1 | tee -a "$LOG_FILE"

TASK_END=$(date +%s)
echo "  ✅ 完成 ($((TASK_END - TASK_START))s)" | tee -a "$LOG_FILE"

# ---- 变体 2: 去掉 utility scoring ----
COUNT=$((COUNT + 1))
echo "" | tee -a "$LOG_FILE"
echo "--- [${COUNT}/${TOTAL}] w/o Utility (只保留 Groundness) ---" | tee -a "$LOG_FILE"
echo "  开始: $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"

TASK_START=$(date +%s)
python "${EVAL_RETRIEVE}" \
    --model_name "${MODEL}" \
    --input_file "${INPUT_FILE}" \
    --output_file "${RESULT_DIR}/popqa_ar_our_no_ut.json" \
    --max_new_tokens "${MAX_TOKENS}" \
    --ndocs "${NDOCS}" \
    --metric match \
    --use_groundness \
    2>&1 | tee -a "$LOG_FILE"

TASK_END=$(date +%s)
echo "  ✅ 完成 ($((TASK_END - TASK_START))s)" | tee -a "$LOG_FILE"

# ---- 变体 3: 去掉所有 scoring (不排序，直接用第一个) ----
COUNT=$((COUNT + 1))
echo "" | tee -a "$LOG_FILE"
echo "--- [${COUNT}/${TOTAL}] w/o All Scoring (无排序) ---" | tee -a "$LOG_FILE"
echo "  开始: $(date '+%H:%M:%S')" | tee -a "$LOG_FILE"

TASK_START=$(date +%s)
python "${EVAL_RETRIEVE}" \
    --model_name "${MODEL}" \
    --input_file "${INPUT_FILE}" \
    --output_file "${RESULT_DIR}/popqa_ar_our_no_score.json" \
    --max_new_tokens "${MAX_TOKENS}" \
    --ndocs "${NDOCS}" \
    --metric match \
    2>&1 | tee -a "$LOG_FILE"

TASK_END=$(date +%s)
echo "  ✅ 完成 ($((TASK_END - TASK_START))s)" | tee -a "$LOG_FILE"

# ==================================================================
# 汇总
# ==================================================================
END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo " 消融实验完成! 总耗时: $((TOTAL_ELAPSED / 60))m $((TOTAL_ELAPSED % 60))s" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "--- PopQA always_retrieve 消融结果 (1,399 条) ---" | tee -a "$LOG_FILE"
printf "%-25s %-10s\n" "Variant" "Score" | tee -a "$LOG_FILE"
printf "%-25s %-10s\n" "───────────────────────" "────────" | tee -a "$LOG_FILE"

for variant in "popqa_ar_our:Full (G+U)" "popqa_ar_our_no_grd:w/o Groundness" "popqa_ar_our_no_ut:w/o Utility" "popqa_ar_our_no_score:w/o All Scoring"; do
    file="${variant%%:*}"
    label="${variant##*:}"
    fpath="${RESULT_DIR}/${file}.json"
    score=$(python -c "import json;d=json.load(open('${fpath}'));print(f\"{d['metric_mean']*100:.2f}%\")" 2>/dev/null || echo "N/A")
    printf "%-25s %-10s\n" "$label" "$score" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "论文参考 (Table 3): Full=54.9%, w/o G=52.1%, w/o U=51.3%" | tee -a "$LOG_FILE"
