#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Self-RAG 批量评测脚本 (no_retrieval 模式专用)

原始 run_short_form.py 逐条推理 (batch_size=1)，速度极慢 (~2.5s/sample)。
本脚本将所有 prompt 一次性交给 vLLM 批量推理，充分利用 GPU 并行能力，
速度提升 10-50 倍。

用法:
    python run_eval_batch.py \
        --model_name outputs/generator_llama2_7b \
        --input_file data/eval/popqa_longtail_w_gs.jsonl \
        --output_file results/popqa_our.json \
        --max_new_tokens 100 --metric match
"""

import json
import argparse
import numpy as np
import time
import sys
import os

# 确保能导入同目录下的 utils 和 metrics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# --- 从 utils.py 导入 ---
from utils import (
    PROMPT_DICT, TASK_INST, load_jsonlines, control_tokens,
)

# --- 从 metrics.py 导入 ---
from metrics import match


def postprocess_answer(answer):
    """去除 special tokens 和多余字符"""
    for token in control_tokens:
        answer = answer.replace(token, "")
    for tok in ["</s>", "<|endoftext|>"]:
        answer = answer.replace(tok, "")
    answer = answer.replace("\n", "")
    return answer.strip()


def preprocess_input_data(dataset, task=None):
    """预处理输入数据，兼容原始和已预处理的数据格式"""
    new_data = []
    instruction = TASK_INST.get(task, None)

    for item in dataset:
        if task == "arc_c":
            # 如果 instruction 已存在（预处理过的数据），直接使用
            if "instruction" in item:
                if "answers" not in item and "answerKey" in item:
                    item["answers"] = [item["answerKey"]]
            elif "choices" in item:
                choices = item["choices"]
                if isinstance(choices, dict) and "label" in choices:
                    # 原始 ARC 格式: choices = {"label": [...], "text": [...]}
                    answer_labels = {}
                    for i in range(len(choices["label"])):
                        answer_key = choices["label"][i]
                        text = choices["text"][i]
                        if answer_key == "1":
                            answer_labels["A"] = text
                        elif answer_key == "2":
                            answer_labels["B"] = text
                        elif answer_key == "3":
                            answer_labels["C"] = text
                        elif answer_key == "4":
                            answer_labels["D"] = text
                        if answer_key in ["A", "B", "C", "D"]:
                            answer_labels[answer_key] = text

                    if "D" not in answer_labels:
                        answer_labels["D"] = ""
                    choices_str = "\nA: {0}\nB: {1}\nC: {2}\nD: {3}".format(
                        answer_labels.get("A", ""),
                        answer_labels.get("B", ""),
                        answer_labels.get("C", ""),
                        answer_labels.get("D", ""),
                    )
                    if "E" in answer_labels:
                        choices_str += "\nE: {}".format(answer_labels["E"])
                    item["instruction"] = instruction + "\n\n### Input:\n" + item["question"] + choices_str
                elif isinstance(choices, str):
                    # 已预处理: choices 是字符串
                    item["instruction"] = instruction + "\n\n### Input:\n" + item["question"] + choices
                else:
                    item["instruction"] = instruction + "\n\n### Input:\n" + item["question"]
                item["answers"] = [item.get("answerKey", item.get("answer", ""))]
            else:
                # 没有 choices 字段，直接用 question
                prompt = instruction + "\n\n### Input:\n" + item.get("question", "")
                item["instruction"] = prompt
                if "answers" not in item and "answerKey" in item:
                    item["answers"] = [item["answerKey"]]
        else:
            if "instruction" not in item:
                prompt = (instruction + "\n\n## Input:\n\n" + item["question"]
                          if instruction else item["question"])
                item["instruction"] = prompt
        new_data.append(item)

    return new_data


def main():
    parser = argparse.ArgumentParser(description="Self-RAG Batch Evaluation (no_retrieval)")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--metric", type=str, default="match", choices=["match"])
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="half")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--download_dir", type=str, default=".cache")
    args = parser.parse_args()

    print(f"=" * 60)
    print(f"  Self-RAG Batch Evaluation (no_retrieval)")
    print(f"  Model: {args.model_name}")
    print(f"  Input: {args.input_file}")
    print(f"  max_new_tokens: {args.max_new_tokens}")
    print(f"=" * 60)

    # --- 1. 加载数据 ---
    t0 = time.time()
    if args.input_file.endswith(".json"):
        input_data = json.load(open(args.input_file))
    else:
        input_data = load_jsonlines(args.input_file)

    # 调试：打印第一条原始数据的字段
    if input_data:
        print(f"  [DEBUG] 第一条数据 keys: {list(input_data[0].keys())}")
        for k in ["instruction", "question", "answers", "answer", "answerKey", "choices"]:
            if k in input_data[0]:
                val = input_data[0][k]
                val_str = str(val)[:100] + "..." if len(str(val)) > 100 else str(val)
                print(f"  [DEBUG]   {k}: {val_str}")

    input_data = preprocess_input_data(input_data, task=args.task)
    print(f"[1/4] 数据加载完成: {len(input_data)} 条 ({time.time()-t0:.1f}s)")

    # --- 2. 构建所有 prompt ---
    t1 = time.time()
    all_prompts = []
    all_answers = []
    for row in input_data:
        prompt = PROMPT_DICT["prompt_no_input"].format_map(row)
        # no_retrieval 模式：直接追加 [No Retrieval]
        prompt += "[No Retrieval]"
        all_prompts.append(prompt)
        # 收集 ground truth
        if "answers" not in row and "answer" in row:
            row["answers"] = [row["answer"]] if isinstance(row["answer"], str) else row["answer"]
        all_answers.append(row.get("answers", []))
    print(f"[2/4] Prompt 构建完成: {len(all_prompts)} 条 ({time.time()-t1:.1f}s)")

    # --- 3. 加载模型 + 批量推理 ---
    t2 = time.time()
    print(f"[3/4] 加载模型...")
    model = LLM(
        model=args.model_name,
        download_dir=args.download_dir,
        dtype=args.dtype,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
    )

    print(f"     模型加载耗时: {time.time()-t2:.1f}s")
    print(f"     开始批量推理 {len(all_prompts)} 条...")
    t3 = time.time()
    outputs = model.generate(all_prompts, sampling_params)
    inference_time = time.time() - t3
    print(f"     推理完成! 耗时: {inference_time:.1f}s ({inference_time/len(all_prompts)*1000:.1f} ms/sample)")

    # --- 4. 计算指标 ---
    t4 = time.time()
    preds = []
    metric_results = []
    for i, output in enumerate(outputs):
        pred_text = output.outputs[0].text
        pred_text = postprocess_answer(pred_text)

        # 特殊处理
        if pred_text and (pred_text[0] == "#" or pred_text[0] == ":"):
            pred_text = pred_text[1:]

        # PubHealth / FEVER 特殊处理
        if "SUPPORTS" in pred_text:
            pred_text_for_metric = "true"
        elif "REFUTES" in pred_text:
            pred_text_for_metric = "false"
        else:
            pred_text_for_metric = pred_text

        preds.append(pred_text)

        if args.metric == "match":
            score = match(pred_text_for_metric, all_answers[i])
        else:
            raise NotImplementedError(f"Metric {args.metric} not implemented")
        metric_results.append(score)

    mean_score = np.mean(metric_results)
    print(f"[4/4] 评测完成 ({time.time()-t4:.1f}s)")
    print(f"")
    print(f"{'='*60}")
    print(f"  结果: {args.metric} = {mean_score:.4f} ({mean_score*100:.2f}%)")
    print(f"  样本数: {len(metric_results)}")
    print(f"  总耗时: {time.time()-t0:.1f}s")
    print(f"{'='*60}")

    # --- 保存结果 ---
    final_results = {
        "model": args.model_name,
        "input_file": args.input_file,
        "metric": args.metric,
        "metric_mean": float(mean_score),
        "num_samples": len(metric_results),
        "inference_time_seconds": inference_time,
        "ms_per_sample": inference_time / len(all_prompts) * 1000,
        "preds": preds,
        "metric_results": metric_results,
    }
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    print(f"结果已保存: {args.output_file}")


if __name__ == "__main__":
    main()
