"""
评测结果汇总脚本
用途：汇总所有评测结果，生成对比表格，计算与论文数字的差异

在集群上执行：
  python scripts/eval/summarize_results.py
"""

import argparse
import json
import os
import glob


# 论文报告的原始数字（Table 1 + Table 2）
PAPER_RESULTS = {
    "selfrag_7b": {
        "PopQA": 54.9,
        "TriviaQA": 68.5,
        "PubHealth": 72.4,
        "ARC-Challenge": 67.3,
        "ASQA_EM": 31.1,
    },
    "selfrag_13b": {
        "PopQA": 55.8,
        "TriviaQA": 69.3,
        "PubHealth": 72.2,
        "ARC-Challenge": 72.4,
    },
    "llama2_7b": {
        "PopQA": 20.2,
        "TriviaQA": 55.6,
        "PubHealth": 34.2,
        "ARC-Challenge": 46.4,
    },
    "llama2_7b_rag": {
        "PopQA": 36.7,
        "TriviaQA": 56.6,
        "PubHealth": 43.7,
        "ARC-Challenge": 46.5,
    },
}


def load_result(filepath):
    """加载评测结果文件，提取核心指标"""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        # 尝试提取不同格式的指标
        if isinstance(data, dict):
            for key in ["accuracy", "score", "em", "match", "metric", "result"]:
                if key in data:
                    val = data[key]
                    return val * 100 if isinstance(val, float) and val <= 1.0 else val
            # 嵌套格式
            if "results" in data and isinstance(data["results"], dict):
                for key in data["results"]:
                    return data["results"][key]
        return None
    except Exception as e:
        print(f"  [WARN] 无法解析 {filepath}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results",
                        help="评测结果根目录")
    parser.add_argument("--output_file", type=str, default="results/reproduction_results.json",
                        help="汇总结果输出文件")
    args = parser.parse_args()

    tasks = ["PopQA", "TriviaQA", "PubHealth", "ARC-Challenge", "ASQA", "FactScore"]
    task_files = {
        "PopQA": "popqa_results.json",
        "TriviaQA": "triviaqa_results.json",
        "PubHealth": "pubhealth_results.json",
        "ARC-Challenge": "arc_results.json",
        "ASQA": "asqa_results.json",
        "FactScore": "factscore_results.json",
    }

    # 发现所有评测结果目录
    eval_dirs = glob.glob(os.path.join(args.results_dir, "eval_*"))
    if not eval_dirs:
        print("[ERROR] 未发现评测结果目录")
        print(f"  请确认 {args.results_dir}/ 下有 eval_* 目录")
        return

    # 收集结果
    all_results = {}
    for eval_dir in sorted(eval_dirs):
        model_tag = os.path.basename(eval_dir).replace("eval_", "")
        results = {}
        for task, filename in task_files.items():
            filepath = os.path.join(eval_dir, filename)
            score = load_result(filepath)
            if score is not None:
                results[task] = round(score, 1) if isinstance(score, float) else score
        if results:
            all_results[model_tag] = results

    # 打印对比表格
    print("\n" + "=" * 90)
    print(" Self-RAG 评测结果汇总")
    print("=" * 90)

    # 表头
    header = f"{'Model':<25s}"
    for task in tasks:
        header += f"  {task:>12s}"
    print(header)
    print("-" * 90)

    # 论文数字
    for paper_model, paper_scores in PAPER_RESULTS.items():
        row = f"{'[Paper] ' + paper_model:<25s}"
        for task in tasks:
            val = paper_scores.get(task, "—")
            row += f"  {str(val):>12s}"
        print(row)
    print("-" * 90)

    # 我们的结果
    for model_tag, results in all_results.items():
        row = f"{'[Ours] ' + model_tag:<25s}"
        for task in tasks:
            val = results.get(task, "—")
            row += f"  {str(val):>12s}"
        print(row)

    # 差异分析
    if "official_7b" in all_results or "our_7b" in all_results:
        print("\n" + "=" * 90)
        print(" 复现差异分析 (Ours - Paper)")
        print("=" * 90)
        compare_tag = "our_7b" if "our_7b" in all_results else "official_7b"
        for task in tasks:
            paper_val = PAPER_RESULTS.get("selfrag_7b", {}).get(task)
            our_val = all_results.get(compare_tag, {}).get(task)
            if paper_val is not None and our_val is not None:
                diff = our_val - paper_val
                emoji = "✅" if abs(diff) < 3 else ("⚠️" if abs(diff) < 5 else "❌")
                print(f"  {task:<20s}: Paper={paper_val:>6.1f}, Ours={our_val:>6.1f}, Diff={diff:>+6.1f} {emoji}")

    # 保存 JSON
    output = {
        "paper_results": PAPER_RESULTS,
        "our_results": all_results,
    }
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] 结果已保存到: {args.output_file}")


if __name__ == "__main__":
    main()
