#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Self-RAG 批量评测脚本 (always_retrieve 模式)

在 always_retrieve 模式下，对每个 query 使用 top-N 检索段落分别生成答案，
通过 Relevance / Groundedness / Utility 三维评分选出最佳段落的答案。

与 run_eval_batch.py (no_retrieval) 的区别:
  - 每个 sample 生成 N 个候选（N = 段落数）
  - 需要 logprobs 来计算评分（vLLM 0.5.5 限制 max=20，足够覆盖 special tokens）
  - 最终输出 = 最高评分段落的生成结果

用法:
    python run_eval_batch_retrieve.py \
        --model_name outputs/generator_llama2_7b \
        --input_file data/eval/popqa_longtail_w_gs.jsonl \
        --output_file results/popqa_retrieve_our.json \
        --max_new_tokens 100 --ndocs 5
"""

import json
import argparse
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from utils import (
    PROMPT_DICT, TASK_INST, load_jsonlines, control_tokens,
    load_special_tokens, postprocess,
)
from metrics import match


def postprocess_answer(answer):
    """去除 special tokens"""
    for token in control_tokens:
        answer = answer.replace(token, "")
    for tok in ["</s>", "<|endoftext|>"]:
        answer = answer.replace(tok, "")
    answer = answer.replace("\n", "")
    return answer.strip()


def preprocess_input_data(dataset, task=None):
    """预处理（与 run_short_form.py 一致）"""
    new_data = []
    instruction = TASK_INST.get(task, None)
    for item in dataset:
        if "instruction" not in item:
            if task == "arc_c":
                # 已预处理格式
                item["instruction"] = instruction + "\n\n### Input:\n" + item.get("question", "")
                if "answers" not in item and "answerKey" in item:
                    item["answers"] = [item["answerKey"]]
            else:
                prompt = (instruction + "\n\n## Input:\n\n" + item["question"]
                          if instruction else item["question"])
                item["instruction"] = prompt
        new_data.append(item)
    return new_data


def score_passage(pred_output, rel_tokens, grd_tokens, ut_tokens,
                  w_rel=1.0, w_sup=1.0, w_use=0.5, use_seqscore=False):
    """计算单个段落生成结果的综合评分"""
    pred_token_ids = pred_output.token_ids
    pred_log_probs = pred_output.logprobs
    seq_score = pred_output.cumulative_logprob / max(len(pred_token_ids), 1)

    # --- Relevance ---
    relevance_scores = {}
    for tok, tok_id in rel_tokens.items():
        if pred_log_probs and len(pred_log_probs) > 0:
            prob = pred_log_probs[0].get(tok_id, -100)
            if hasattr(prob, 'logprob'):
                prob = prob.logprob
            relevance_scores[tok] = np.exp(float(prob))
        else:
            relevance_scores[tok] = 0.0

    rel_sum = sum(relevance_scores.values())
    relevance_score = relevance_scores.get("[Relevant]", 0.0) / max(rel_sum, 1e-10)

    # --- Groundedness ---
    ground_score = 0.0
    if grd_tokens is not None and pred_log_probs:
        grd_scores = {}
        # 找到 groundedness token 首次出现的位置
        grd_idx = None
        for tok_idx, tok_id in enumerate(pred_token_ids):
            if tok_id in list(grd_tokens.values()):
                grd_idx = tok_idx
                break
        if grd_idx is not None and grd_idx < len(pred_log_probs):
            for token, token_id in grd_tokens.items():
                prob = pred_log_probs[grd_idx].get(token_id, -100)
                if hasattr(prob, 'logprob'):
                    prob = prob.logprob
                grd_scores[token] = np.exp(float(prob))
            if len(grd_scores) == 3:
                gt_sum = sum(grd_scores.values())
                ground_score = (grd_scores.get("[Fully supported]", 0) / max(gt_sum, 1e-10)) + \
                               0.5 * (grd_scores.get("[Partially supported]", 0) / max(gt_sum, 1e-10))

    # --- Utility ---
    utility_score = 0.0
    if ut_tokens is not None and pred_log_probs:
        ut_scores_dict = {}
        ut_idx = None
        for tok_idx, tok_id in enumerate(pred_token_ids):
            if tok_id in list(ut_tokens.values()):
                ut_idx = tok_idx
                break
        if ut_idx is not None and ut_idx < len(pred_log_probs):
            for token, token_id in ut_tokens.items():
                prob = pred_log_probs[ut_idx].get(token_id, -100)
                if hasattr(prob, 'logprob'):
                    prob = prob.logprob
                ut_scores_dict[token] = np.exp(float(prob))
            if len(ut_scores_dict) == 5:
                ut_sum = sum(ut_scores_dict.values())
                ut_weights = [-1, -0.5, 0, 0.5, 1]
                utility_score = sum(
                    ut_weights[i] * (ut_scores_dict.get("[Utility:{}]".format(i+1), 0) / max(ut_sum, 1e-10))
                    for i in range(5)
                )

    # --- 综合评分 ---
    if use_seqscore:
        final_score = np.exp(seq_score) + w_rel * relevance_score + w_sup * ground_score + w_use * utility_score
    else:
        final_score = w_rel * relevance_score + w_sup * ground_score + w_use * utility_score

    return final_score


def main():
    parser = argparse.ArgumentParser(description="Self-RAG Batch Evaluation (always_retrieve)")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--metric", type=str, default="match")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--ndocs", type=int, default=5, help="Number of top passages per sample")
    parser.add_argument("--dtype", type=str, default="half")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--download_dir", type=str, default=".cache")
    parser.add_argument("--use_groundness", action="store_true")
    parser.add_argument("--use_utility", action="store_true")
    parser.add_argument("--use_seqscore", action="store_true")
    parser.add_argument("--w_rel", type=float, default=1.0)
    parser.add_argument("--w_sup", type=float, default=1.0)
    parser.add_argument("--w_use", type=float, default=0.5)
    args = parser.parse_args()

    print("=" * 60)
    print("  Self-RAG Batch Evaluation (always_retrieve)")
    print(f"  Model: {args.model_name}")
    print(f"  Input: {args.input_file}")
    print(f"  ndocs: {args.ndocs}")
    print(f"  max_new_tokens: {args.max_new_tokens}")
    print("=" * 60)

    # --- 1. 加载数据 ---
    t0 = time.time()
    if args.input_file.endswith(".json"):
        input_data = json.load(open(args.input_file))
    else:
        input_data = load_jsonlines(args.input_file)
    input_data = preprocess_input_data(input_data, task=args.task)
    print(f"[1/5] 数据加载完成: {len(input_data)} 条 ({time.time()-t0:.1f}s)")

    # --- 2. 构建所有 (sample × passage) prompts ---
    t1 = time.time()
    all_prompts = []        # 展平的所有 prompt
    sample_indices = []     # 每个 prompt 对应的 sample index
    passage_counts = []     # 每个 sample 有多少个 passage
    all_answers = []

    for i, row in enumerate(input_data):
        prompt = PROMPT_DICT["prompt_no_input"].format_map(row)
        ctx_key = "ctxs" if "ctxs" in row else "top_contexts"
        evidences = row.get(ctx_key, [])[:args.ndocs]

        if len(evidences) == 0:
            # 无检索段落 → 退回 no_retrieval
            if i == 0:
                print("⚠️  WARNING: 数据中不包含 'ctxs'/'top_contexts' 字段！")
                print("   always_retrieve 模式将退化为 no_retrieval，结果可能不可靠。")
                print("   请确认数据文件包含预检索的段落（ctxs 字段）。")
            all_prompts.append(prompt + "[No Retrieval]")
            sample_indices.append(i)
            passage_counts.append(1)
        else:
            for para in evidences:
                aug_prompt = prompt + "[Retrieval]<paragraph>{}\n{}</paragraph>".format(
                    para.get("title", ""), para.get("text", ""))
                all_prompts.append(aug_prompt)
                sample_indices.append(i)
            passage_counts.append(len(evidences))

        # Ground truth
        if "answers" not in row and "answer" in row:
            row["answers"] = [row["answer"]] if isinstance(row["answer"], str) else row["answer"]
        all_answers.append(row.get("answers", []))

    total_prompts = len(all_prompts)
    print(f"[2/5] Prompt 构建完成: {len(input_data)} samples × ~{args.ndocs} docs = {total_prompts} prompts ({time.time()-t1:.1f}s)")

    # --- 3. 加载模型 ---
    t2 = time.time()
    print("[3/5] 加载模型...")
    model = LLM(
        model=args.model_name,
        download_dir=args.download_dir,
        dtype=args.dtype,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(
        tokenizer, use_grounding=args.use_groundness, use_utility=args.use_utility)

    print(f"     模型加载耗时: {time.time()-t2:.1f}s")

    # --- 4. 批量推理 ---
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
        logprobs=20,
    )

    print(f"[4/5] 开始批量推理 {total_prompts} 条...")
    t3 = time.time()
    outputs = model.generate(all_prompts, sampling_params)
    inference_time = time.time() - t3
    print(f"     推理完成! 耗时: {inference_time:.1f}s ({inference_time/total_prompts*1000:.1f} ms/prompt)")

    # --- 5. 评分 + 选最佳答案 ---
    t4 = time.time()
    preds = []
    metric_results = []

    flat_idx = 0
    for sample_idx in range(len(input_data)):
        n_passages = passage_counts[sample_idx]
        best_score = -float('inf')
        best_pred = ""

        for p in range(n_passages):
            output = outputs[flat_idx]
            pred_text = output.outputs[0].text

            if n_passages == 1:
                # no_retrieval fallback
                best_pred = postprocess_answer(pred_text)
            else:
                score = score_passage(
                    output.outputs[0],
                    rel_tokens, grd_tokens, ut_tokens,
                    w_rel=args.w_rel, w_sup=args.w_sup, w_use=args.w_use,
                    use_seqscore=args.use_seqscore,
                )
                if score > best_score:
                    best_score = score
                    best_pred = postprocess_answer(pred_text)
            flat_idx += 1

        # 特殊字符处理
        if best_pred and (best_pred[0] == "#" or best_pred[0] == ":"):
            best_pred = best_pred[1:]

        preds.append(best_pred)

        # 计算指标
        if args.metric == "match":
            score = match(best_pred, all_answers[sample_idx])
        else:
            raise NotImplementedError
        metric_results.append(score)

        if (sample_idx + 1) % 1000 == 0:
            print(f"     进度: {sample_idx+1}/{len(input_data)}, 当前 match = {np.mean(metric_results):.4f}")

    mean_score = np.mean(metric_results)
    print(f"[5/5] 评测完成 ({time.time()-t4:.1f}s)")
    print()
    print("=" * 60)
    print(f"  结果: {args.metric} = {mean_score:.4f} ({mean_score*100:.2f}%)")
    print(f"  样本数: {len(metric_results)}")
    print(f"  总耗时: {time.time()-t0:.1f}s")
    print("=" * 60)

    # --- 保存 ---
    final_results = {
        "model": args.model_name,
        "input_file": args.input_file,
        "mode": "always_retrieve",
        "ndocs": args.ndocs,
        "metric": args.metric,
        "metric_mean": float(mean_score),
        "num_samples": len(metric_results),
        "total_prompts": total_prompts,
        "inference_time_seconds": inference_time,
        "ms_per_sample": inference_time / len(input_data) * 1000,
        "preds": preds,
        "metric_results": metric_results,
    }
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    print(f"结果已保存: {args.output_file}")


if __name__ == "__main__":
    main()
