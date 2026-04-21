"""
Self-RAG 可视化分析脚本
用途：生成所有报告所需图表（300dpi, 统一风格）

执行：python scripts/visualize.py --results_dir results --output_dir figures
"""

import argparse
import json
import os
import glob
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# ---- 全局风格 ----
sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.figsize": (8, 5),
})

COLORS = {
    "selfrag": "#1976D2",
    "vanilla": "#F57C00",
    "rag": "#388E3C",
    "llama3": "#7B1FA2",
    "qwen": "#C62828",
}


def plot_retrieval_trigger_rate(results_dir, output_dir):
    """图1：各任务检索触发率柱状图"""
    # 读取推理结果中的反思 Token 统计
    inference_file = os.path.join(results_dir, "quick_inference_results.json")
    if not os.path.exists(inference_file):
        print("[SKIP] 检索触发率图（需要 quick_inference_results.json）")
        return

    with open(inference_file) as f:
        data = json.load(f)

    # 从评测结果中收集检索触发率
    tasks = ["PubHealth", "ARC-C", "TriviaQA", "PopQA", "ASQA"]
    retrieval_rates = [0.65, 0.72, 0.81, 0.88, 0.95]  # 占位数据，实际由评测脚本生成
    no_retrieval_rates = [1 - r for r in retrieval_rates]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(tasks))
    width = 0.35

    bars1 = ax.bar(x - width / 2, retrieval_rates, width, label="[Retrieval]", color="#1976D2", alpha=0.85)
    bars2 = ax.bar(x + width / 2, no_retrieval_rates, width, label="[No Retrieval]", color="#FF8A65", alpha=0.85)

    ax.set_ylabel("Proportion")
    ax.set_title("Retrieval Trigger Rate Across Tasks")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.legend()
    ax.set_ylim(0, 1.1)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.0%}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "retrieval_trigger_rate.png"), bbox_inches="tight")
    plt.close()
    print("[OK] 检索触发率柱状图")


def plot_model_comparison_radar(results_dir, output_dir):
    """图3：基座模型对比雷达图 (Llama 2 vs Llama 3 vs Qwen)"""
    tasks = ["PubHealth", "ARC-C", "TriviaQA", "PopQA", "ASQA"]

    # 示例数据（实际运行后替换）
    models = {
        "Llama 2 7B": [72.4, 67.3, 68.5, 54.9, 31.1],
        "Llama 3 8B": [75.0, 71.0, 72.0, 58.0, 34.0],  # 预期改进
    }

    angles = np.linspace(0, 2 * np.pi, len(tasks), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    colors = ["#1976D2", "#7B1FA2", "#C62828"]
    for idx, (model_name, scores) in enumerate(models.items()):
        scores_plot = scores + scores[:1]
        ax.plot(angles, scores_plot, "o-", linewidth=2, label=model_name, color=colors[idx])
        ax.fill(angles, scores_plot, alpha=0.15, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(tasks, fontsize=10)
    ax.set_title("Self-RAG: Base Model Comparison", pad=20, fontsize=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison_radar.png"), bbox_inches="tight")
    plt.close()
    print("[OK] 基座模型对比雷达图")


def plot_ablation_bar(results_dir, output_dir):
    """图4：消融实验柱状图"""
    settings = ["Full\nSelf-RAG", "No Reflection\n(Standard RAG)", "Always\nRetrieve", "No\nRetrieval"]
    # 示例数据（PubHealth 上的表现）
    pubhealth = [72.4, 43.7, 68.1, 34.2]
    triviaqa = [68.5, 56.6, 65.2, 55.6]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(settings))
    width = 0.35

    bars1 = ax.bar(x - width / 2, pubhealth, width, label="PubHealth", color="#1976D2", alpha=0.85)
    bars2 = ax.bar(x + width / 2, triviaqa, width, label="TriviaQA", color="#FF8A65", alpha=0.85)

    ax.set_ylabel("Score (%)")
    ax.set_title("Ablation Study: Impact of Self-RAG Components")
    ax.set_xticks(x)
    ax.set_xticklabels(settings)
    ax.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ablation_bar.png"), bbox_inches="tight")
    plt.close()
    print("[OK] 消融实验柱状图")


def plot_topk_sensitivity(results_dir, output_dir):
    """图5：Top-K 敏感性折线图"""
    k_values = [1, 3, 5, 10, 20]
    # 示例数据
    pubhealth = [65.0, 70.1, 72.4, 71.8, 70.5]
    triviaqa = [60.2, 65.8, 68.5, 68.1, 67.3]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, pubhealth, "o-", linewidth=2, markersize=8, label="PubHealth", color="#1976D2")
    ax.plot(k_values, triviaqa, "s--", linewidth=2, markersize=8, label="TriviaQA", color="#F57C00")

    ax.set_xlabel("Top-K Retrieved Documents")
    ax.set_ylabel("Score (%)")
    ax.set_title("Sensitivity to Number of Retrieved Documents (Top-K)")
    ax.set_xticks(k_values)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "topk_sensitivity.png"), bbox_inches="tight")
    plt.close()
    print("[OK] Top-K 敏感性折线图")


def plot_reproduction_comparison(results_dir, output_dir):
    """图6：复现 vs 论文原始结果对比"""
    tasks = ["PopQA", "TriviaQA", "PubHealth", "ARC-C"]

    paper = [54.9, 68.5, 72.4, 67.3]
    ours = [53.5, 67.8, 71.9, 66.5]  # 示例数据

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(tasks))
    width = 0.3

    bars1 = ax.bar(x - width / 2, paper, width, label="Paper (Original)", color="#1976D2", alpha=0.85)
    bars2 = ax.bar(x + width / 2, ours, width, label="Ours (Reproduced)", color="#F57C00", alpha=0.85)

    ax.set_ylabel("Score (%)")
    ax.set_title("Reproduction: Our Results vs. Paper's Reported Numbers")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.legend()

    # 标注差异
    for i, (p, o) in enumerate(zip(paper, ours)):
        diff = o - p
        color = "#388E3C" if diff >= 0 else "#D32F2F"
        ax.text(x[i] + width / 2, o + 0.5, f"{diff:+.1f}", ha="center", fontsize=9, color=color, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reproduction_comparison.png"), bbox_inches="tight")
    plt.close()
    print("[OK] 复现对比图")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--output_dir", default="figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 50)
    print(" 生成可视化图表")
    print("=" * 50)

    plot_retrieval_trigger_rate(args.results_dir, args.output_dir)
    plot_model_comparison_radar(args.results_dir, args.output_dir)
    plot_ablation_bar(args.results_dir, args.output_dir)
    plot_topk_sensitivity(args.results_dir, args.output_dir)
    plot_reproduction_comparison(args.results_dir, args.output_dir)

    print(f"\n所有图表已保存到: {args.output_dir}/")
    print("注意：当前使用示例数据，请在获得真实评测结果后更新数据！")


if __name__ == "__main__":
    main()
