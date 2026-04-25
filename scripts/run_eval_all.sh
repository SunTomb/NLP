#!/bin/bash
# =============================================================================
# Self-RAG 全面评测脚本 — 3 任务 × 3 模型 = 9 次评测 (批量推理版)
# 用法: CUDA_VISIBLE_DEVICES=1 bash scripts/run_eval_all.sh
# 预计耗时: ~30-60 分钟 | 显存: ~35-40 GB (单卡, vLLM 自动管理)
# =============================================================================

set -eo pipefail  # 任意命令失败即停止（含管道内命令）

# --- 配置 ---
EVAL_SCRIPT="self-rag/retrieval_lm/run_eval_batch.py"
RESULT_DIR="results"
LOG_DIR="logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/eval_all_${TIMESTAMP}.log"

# 模型列表
declare -A MODELS
MODELS["our"]="outputs/generator_llama2_7b"
MODELS["official"]="models/selfrag_llama2_7b"
MODELS["llama2"]="models/Llama-2-7b-hf"

# 任务定义: task_key|input_file|max_new_tokens|metric|extra_args
# 文件名以实际 data/eval/ 目录下的为准
TASKS=(
    "popqa|data/eval/popqa_longtail_w_gs.jsonl|100|match|"
    "arc|data/eval/arc_challenge_processed.jsonl|50|match|--task arc_c"
    "triviaqa|data/eval/triviaqa_test_w_gs.jsonl|100|match|"
)

# --- 初始化 ---
mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

echo "============================================================" | tee -a "$LOG_FILE"
echo " Self-RAG Evaluation — $(date)" | tee -a "$LOG_FILE"
echo " 3 tasks × 3 models = 9 evaluations (batch inference)" | tee -a "$LOG_FILE"
echo " Log: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

TOTAL=9
COUNT=0
START_TIME=$(date +%s)

# --- 主循环 ---
for model_key in our official llama2; do
    model_path="${MODELS[$model_key]}"
    echo "" | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"
    echo " 模型: ${model_key} (${model_path})" | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"

    for task_entry in "${TASKS[@]}"; do
        IFS='|' read -r task_key input_file max_tokens metric extra_args <<< "$task_entry"
        output_file="${RESULT_DIR}/${task_key}_${model_key}.json"
        COUNT=$((COUNT + 1))

        echo "" | tee -a "$LOG_FILE"
        echo "--- [${COUNT}/${TOTAL}] ${task_key} × ${model_key} ---" | tee -a "$LOG_FILE"
        echo "  输入: ${input_file}" | tee -a "$LOG_FILE"
        echo "  输出: ${output_file}" | tee -a "$LOG_FILE"
        echo "  max_new_tokens: ${max_tokens}" | tee -a "$LOG_FILE"
        echo "  开始时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"

        TASK_START=$(date +%s)

        python "${EVAL_SCRIPT}" \
            --model_name "${model_path}" \
            --input_file "${input_file}" \
            --max_new_tokens "${max_tokens}" \
            --output_file "${output_file}" \
            --metric "${metric}" \
            ${extra_args} \
            2>&1 | tee -a "$LOG_FILE"

        TASK_END=$(date +%s)
        TASK_ELAPSED=$(( TASK_END - TASK_START ))
        echo "  ✅ 完成! 耗时: $((TASK_ELAPSED / 60))m $((TASK_ELAPSED % 60))s" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
    done
done

# --- 汇总结果 ---
END_TIME=$(date +%s)
TOTAL_ELAPSED=$(( END_TIME - START_TIME ))

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo " 全部评测完成! 总耗时: $((TOTAL_ELAPSED / 3600))h $((TOTAL_ELAPSED % 3600 / 60))m" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "--- 结果汇总 ---" | tee -a "$LOG_FILE"

printf "%-15s %-12s %-12s %-12s\n" "Task" "Our" "Official" "Llama2" | tee -a "$LOG_FILE"
printf "%-15s %-12s %-12s %-12s\n" "───────────" "──────────" "──────────" "──────────" | tee -a "$LOG_FILE"

for task_entry in "${TASKS[@]}"; do
    IFS='|' read -r task_key input_file max_tokens metric extra_args <<< "$task_entry"
    
    scores=""
    for model_key in our official llama2; do
        result_file="${RESULT_DIR}/${task_key}_${model_key}.json"
        if [ -f "$result_file" ]; then
            score=$(python -c "import json; d=json.load(open('${result_file}')); print(f\"{d['metric_mean']*100:.2f}%\")" 2>/dev/null || echo "ERROR")
        else
            score="N/A"
        fi
        scores="${scores} $(printf '%-12s' "$score")"
    done
    printf "%-15s %s\n" "$task_key" "$scores" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "详细结果文件: ${RESULT_DIR}/" | tee -a "$LOG_FILE"
echo "完整日志: ${LOG_FILE}" | tee -a "$LOG_FILE"
