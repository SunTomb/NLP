"""
Self-RAG 快速推理验证脚本
用途：使用官方预训练模型验证推理流程，观察反思 Token 行为

在集群上执行：
  conda activate selfrag
  python scripts/quick_inference.py --model_path models/selfrag_llama2_7b

需要 1×A40/A100（约 16GB 显存）
"""

import argparse
import json
import os
import re
from collections import Counter


def format_prompt(input_text, paragraph=None):
    """格式化 Self-RAG 的输入 prompt"""
    prompt = f"### Instruction:\n{input_text}\n\n### Response:\n"
    if paragraph is not None:
        prompt += f"[Retrieval]<paragraph>{paragraph}</paragraph>"
    return prompt


def analyze_reflection_tokens(text):
    """分析输出中的反思 Token"""
    tokens = {
        "retrieval": [],
        "relevance": [],
        "support": [],
        "utility": [],
    }

    # 检索决策 Token
    if "[No Retrieval]" in text:
        tokens["retrieval"].append("No Retrieval")
    if "[Retrieval]" in text:
        tokens["retrieval"].append("Retrieval")
    if "[Continue to Use Evidence]" in text:
        tokens["retrieval"].append("Continue to Use Evidence")

    # 相关性 Token
    if "[Relevant]" in text:
        tokens["relevance"].append("Relevant")
    if "[Irrelevant]" in text:
        tokens["relevance"].append("Irrelevant")

    # 支持度 Token
    if "[Fully supported]" in text:
        tokens["support"].append("Fully supported")
    if "[Partially supported]" in text:
        tokens["support"].append("Partially supported")
    if "[No support / Contradictory]" in text:
        tokens["support"].append("No support / Contradictory")

    # 有用性 Token
    for i in range(1, 6):
        if f"[Utility:{i}]" in text:
            tokens["utility"].append(f"Utility:{i}")

    return tokens


def strip_reflection_tokens(text):
    """去除反思 Token，保留纯文本回答"""
    patterns = [
        r"\[No Retrieval\]", r"\[Retrieval\]", r"\[Continue to Use Evidence\]",
        r"\[Relevant\]", r"\[Irrelevant\]",
        r"\[Fully supported\]", r"\[Partially supported\]", r"\[No support / Contradictory\]",
        r"\[Utility:[1-5]\]",
        r"<paragraph>.*?</paragraph>",
        r"</s>",
    ]
    clean = text
    for p in patterns:
        clean = re.sub(p, "", clean)
    return clean.strip()


def main():
    parser = argparse.ArgumentParser(description="Self-RAG Quick Inference")
    parser.add_argument("--model_path", type=str, default="models/selfrag_llama2_7b",
                        help="Self-RAG 模型路径")
    parser.add_argument("--output_file", type=str, default="results/quick_inference_results.json",
                        help="输出结果文件")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85,
                        help="GPU 显存使用比例")
    args = parser.parse_args()

    print("=" * 60)
    print(" Self-RAG 快速推理验证")
    print("=" * 60)

    # ---- 加载模型 ----
    print(f"\n[1] 加载模型: {args.model_path}")
    from vllm import LLM, SamplingParams

    model = LLM(
        args.model_path,
        dtype="half",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=2048,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=200,
        skip_special_tokens=False,
    )
    print("[OK] 模型加载完成")

    # ---- 测试用例 ----
    test_cases = [
        {
            "name": "简单事实问题（不需要检索）",
            "query": "What is 2+2?",
            "paragraph": None,
            "expected_behavior": "模型应输出 [No Retrieval]，直接回答",
        },
        {
            "name": "常识判断（不需要检索）",
            "query": "Leave odd one out: twitter, instagram, whatsapp.",
            "paragraph": None,
            "expected_behavior": "模型应输出 [No Retrieval]，进行常识推理",
        },
        {
            "name": "知识问题（可能触发检索）",
            "query": "Can you tell me the difference between llamas and alpacas?",
            "paragraph": None,
            "expected_behavior": "模型应输出 [Retrieval]，表示需要外部知识",
        },
        {
            "name": "知识问题 + 检索段落",
            "query": "Can you tell me the difference between llamas and alpacas?",
            "paragraph": "The alpaca (Lama pacos) is a species of South American camelid mammal. It is similar to, and often confused with, the llama. Alpacas are considerably smaller than llamas, and unlike llamas, they were not bred to be working animals, but were bred specifically for their fiber.",
            "expected_behavior": "模型应输出 [Relevant] + [Fully supported] + 高 Utility",
        },
        {
            "name": "科学问题（需要检索）",
            "query": "What is overfitting in machine learning?",
            "paragraph": "In statistics, overfitting is the production of an analysis that corresponds too closely or exactly to a particular set of data, and may therefore fail to fit additional data or predict future observations reliably.",
            "expected_behavior": "模型应利用文档生成准确回答",
        },
        {
            "name": "中文问题测试",
            "query": "中国的首都是哪里？",
            "paragraph": None,
            "expected_behavior": "测试中文支持（Llama 2 对中文支持有限）",
        },
        {
            "name": "健康事实核查",
            "query": "Is it true that drinking 8 glasses of water a day is necessary for good health?",
            "paragraph": None,
            "expected_behavior": "测试事实核查场景",
        },
    ]

    # ---- 执行推理 ----
    results = []
    all_tokens = Counter()

    print(f"\n[2] 开始推理 ({len(test_cases)} 个测试用例)...")
    print("-" * 60)

    for i, tc in enumerate(test_cases):
        prompt = format_prompt(tc["query"], tc["paragraph"])
        preds = model.generate([prompt], sampling_params)
        raw_output = preds[0].outputs[0].text

        # 分析反思 Token
        tokens = analyze_reflection_tokens(raw_output)
        clean_answer = strip_reflection_tokens(raw_output)

        # 统计
        for category, token_list in tokens.items():
            for t in token_list:
                all_tokens[t] += 1

        result = {
            "id": i,
            "name": tc["name"],
            "query": tc["query"],
            "has_paragraph": tc["paragraph"] is not None,
            "raw_output": raw_output,
            "clean_answer": clean_answer,
            "reflection_tokens": tokens,
            "expected_behavior": tc["expected_behavior"],
        }
        results.append(result)

        # 打印
        print(f"\n{'='*60}")
        print(f"  测试 {i+1}: {tc['name']}")
        print(f"  Query: {tc['query']}")
        if tc["paragraph"]:
            print(f"  Paragraph: {tc['paragraph'][:80]}...")
        print(f"  预期行为: {tc['expected_behavior']}")
        print(f"  ---")
        print(f"  原始输出: {raw_output}")
        print(f"  纯文本回答: {clean_answer}")
        print(f"  反思 Token: {tokens}")

    # ---- 统计分析 ----
    print(f"\n{'='*60}")
    print(f" 反思 Token 分布统计")
    print(f"{'='*60}")
    for token, count in all_tokens.most_common():
        print(f"  {token:35s}: {count}")

    # 检索触发率
    retrieval_count = sum(1 for r in results if "Retrieval" in r["reflection_tokens"]["retrieval"])
    no_retrieval_count = sum(1 for r in results if "No Retrieval" in r["reflection_tokens"]["retrieval"])
    print(f"\n  检索触发率: {retrieval_count}/{len(results)} ({retrieval_count/len(results)*100:.1f}%)")
    print(f"  不检索率:   {no_retrieval_count}/{len(results)} ({no_retrieval_count/len(results)*100:.1f}%)")

    # ---- 保存结果 ----
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    output = {
        "model": args.model_path,
        "num_test_cases": len(test_cases),
        "token_distribution": dict(all_tokens),
        "results": results,
    }
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] 结果已保存到: {args.output_file}")

    print(f"\n{'='*60}")
    print(f" 推理验证完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
