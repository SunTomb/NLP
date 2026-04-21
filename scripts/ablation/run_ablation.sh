#!/bin/bash
# ============================================================
# 消融实验批量运行脚本
# 在集群上执行:
#   conda activate selfrag
#   bash scripts/ablation/run_ablation.sh
#
# 硬件需求: 1-2×A40 48GB（推理为主，可并行多组）
# ============================================================
set -e

export PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR/self-rag/retrieval_lm"

MODEL="$PROJECT_DIR/outputs/generator_llama2_7b"  # 或官方模型
EVAL_DATA_DIR="$PROJECT_DIR/data/eval"
AB_DIR="$PROJECT_DIR/results/ablation"
mkdir -p "$AB_DIR"

echo "==========================================="
echo " Self-RAG 消融实验"
echo "==========================================="

# 选择评测任务（消融至少在3个任务上跑）
TASKS=("pubhealth" "triviaqa" "popqa")
TASK_FILES=("health_claims_processed.jsonl" "triviaqa_test_w_gs.jsonl" "popqa_longtail_w_gs.jsonl")
TASK_NAMES=("fever" "qa" "qa")
TASK_TOKENS=("50" "100" "100")

run_eval() {
    local model=$1 mode=$2 tag=$3 extra_args=$4
    local out_dir="$AB_DIR/$tag"
    mkdir -p "$out_dir"

    for i in "${!TASKS[@]}"; do
        local task_id="${TASKS[$i]}"
        local input_file="$EVAL_DATA_DIR/${TASK_FILES[$i]}"
        local task_name="${TASK_NAMES[$i]}"
        local max_tokens="${TASK_TOKENS[$i]}"

        [ ! -f "$input_file" ] && echo "  [SKIP] $task_id (数据不存在)" && continue

        echo "  [$tag] $task_id ..."
        python run_short_form.py \
            --model_name "$model" \
            --input_file "$input_file" \
            --mode "$mode" \
            --max_new_tokens "$max_tokens" \
            --threshold 0.2 \
            --output_file "$out_dir/${task_id}_results.json" \
            --metric match \
            --ndocs 10 \
            --dtype half \
            --task "$task_name" \
            $extra_args 2>/dev/null && echo "    [OK]" || echo "    [FAIL]"
    done
}

# ---- Ab1: 标准 RAG（退化，无反思 Token 控制） ----
echo ""
echo "[Ab1] 标准 RAG（无反思 Token 评估）"
run_eval "$MODEL" "always_retrieve" "ab1_no_reflection" ""

# ---- Ab4: 始终检索 ----
echo ""
echo "[Ab4] 始终检索 (always_retrieve)"
run_eval "$MODEL" "always_retrieve" "ab4_always_retrieve" "--use_groundness --use_utility --use_seqscore"

# ---- Full Self-RAG（参照组） ----
echo ""
echo "[Full] Self-RAG (adaptive_retrieval)"
run_eval "$MODEL" "adaptive_retrieval" "full_selfrag" "--use_groundness --use_utility --use_seqscore"

# ---- Ab5: Top-K 敏感性 ----
echo ""
echo "[Ab5] Top-K 敏感性分析"
for K in 1 3 5 10 20; do
    echo "  K=$K ..."
    run_eval "$MODEL" "adaptive_retrieval" "ab5_topk_${K}" "--use_groundness --use_utility --use_seqscore --ndocs $K"
done

# ---- No Retrieval（纯生成） ----
echo ""
echo "[NoRet] 无检索 (no_retrieval)"
run_eval "$MODEL" "no_retrieval" "no_retrieval" ""

echo ""
echo "==========================================="
echo " 消融实验完成！"
echo "==========================================="
echo " 结果保存在: $AB_DIR/"
ls -d "$AB_DIR"/*
echo ""
echo "汇总: python scripts/ablation/summarize_ablation.py"
