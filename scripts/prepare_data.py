"""
Self-RAG 数据准备脚本
===================
解决原始 Google Drive 链接失效问题，从可用来源重建所有训练和评测数据。

功能：
1. 从 Generator 训练数据中合成 Critic 训练数据
2. 从 HuggingFace 下载原始评测数据集并处理为 Self-RAG 格式

用法：
  source /NAS/yesh/NLP/activate.sh
  python scripts/prepare_data.py --task all
  python scripts/prepare_data.py --task critic     # 只生成 Critic 数据
  python scripts/prepare_data.py --task eval       # 只下载评测数据
"""

import argparse
import json
import os
import random
import re
from collections import Counter

random.seed(42)

# ============================================================
#  Critic 训练数据生成
# ============================================================
# 原始 Critic 数据由 GPT-4 标注，格式为 [{instruction, input, output}, ...]
# 我们从 Generator 训练数据中提取反思 Token，合成等价的 Critic 训练数据

CRITIC_PROMPTS = {
    "retrieval_instruction": (
        "When provided with instruction, please evaluate whether seeking additional "
        "information from external sources such as the web (e.g., Wikipedia) aids in "
        "producing a more comprehensive response. Respond with either [Retrieval] or [No Retrieval]."
    ),
    "retrieval_input": "Task instruction: {instruction}",

    "relevance_instruction": (
        "When given instruction and evidence, evaluate whether the evidence is relevant "
        "to the instruction and provides valuable information for generating meaningful "
        "responses.\nUse a rating of [Relevant] to indicate relevance and usefulness, "
        "and [Irrelevant] to indicate irrelevance."
    ),
    "relevance_input": "Task instruction: {instruction}\nEvidence: {evidence}",

    "groundedness_instruction": (
        "You will receive an instruction, evidence, and output, and optional preceding "
        "sentences. Your task is to evaluate if the output is fully supported by the "
        "information provided in the evidence.\n"
        "[Fully supported] - All information in output is supported by the evidence.\n"
        "[Partially supported] - The output is supported by the evidence to some extent, "
        "but there is major information not discussed in the evidence.\n"
        "[No support / Contradictory] - The output completely ignores evidence, is "
        "unrelated to the evidence, or contradicts the evidence."
    ),
    "groundedness_input": "Task instruction: {instruction}\nOutput: {output}\nEvidence: {evidence}",

    "utility_instruction": (
        "Given an instruction and an output, rate whether the response appears to be a "
        "helpful and informative answer to the query, from 1 (lowest) - 5 (highest). "
        "We call this score perceived utility.\n"
        "[Utility:5]: The response provides a complete, highly detailed, and informative "
        "response to the query, fully satisfying the information needs.\n"
        "[Utility:4]: The response mostly fulfills the need in the query.\n"
        "[Utility:3]: The response is acceptable, but some major additions or improvements are needed.\n"
        "[Utility:2]: The response still addresses the main request, but it is not complete.\n"
        "[Utility:1]: The response is barely on-topic or completely irrelevant."
    ),
    "utility_input": "Task instruction: {instruction}\nOutput: {output}",
}

REFLECTION_TOKENS = {
    "retrieval": ["[Retrieval]", "[No Retrieval]", "[Continue to Use Evidence]"],
    "relevance": ["[Relevant]", "[Irrelevant]"],
    "support": ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]"],
    "utility": ["[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"],
}


def extract_reflection_tokens(text):
    """从文本中提取反思 Token"""
    tokens = {}
    for category, token_list in REFLECTION_TOKENS.items():
        for token in token_list:
            if token in text:
                tokens[category] = token
                break
    return tokens


def extract_paragraph(text):
    """从文本中提取段落"""
    match = re.search(r'<paragraph>(.*?)</paragraph>', text, re.DOTALL)
    return match.group(1).strip() if match else None


def strip_reflection_tokens(text):
    """去除反思 Token，保留纯文本"""
    clean = text
    for token_list in REFLECTION_TOKENS.values():
        for token in token_list:
            clean = clean.replace(token, "")
    clean = re.sub(r'<paragraph>.*?</paragraph>', '', clean, flags=re.DOTALL)
    clean = re.sub(r'</s>', '', clean)
    return clean.strip()


def generate_critic_data(generator_data_path, output_path, max_samples=50000):
    """从 Generator 训练数据中合成 Critic 训练数据"""
    print(f"\n[Critic] 从 Generator 数据生成 Critic 训练数据...")
    print(f"  源文件: {generator_data_path}")

    critic_data = []

    with open(generator_data_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            if len(critic_data) >= max_samples * 4:  # 4 种任务
                break
            try:
                entry = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            instruction = entry.get("instruction", "")
            output_text = entry.get("output", "")

            if not instruction or not output_text:
                continue

            # 提取反思 Token
            tokens = extract_reflection_tokens(output_text)
            paragraph = extract_paragraph(output_text)
            clean_output = strip_reflection_tokens(output_text)

            # 1. Retrieval 任务
            if "retrieval" in tokens:
                critic_data.append({
                    "instruction": CRITIC_PROMPTS["retrieval_instruction"],
                    "input": CRITIC_PROMPTS["retrieval_input"].format(instruction=instruction),
                    "output": tokens["retrieval"],
                    "task": "retrieval"
                })

            # 2. Relevance 任务（需要有段落）
            if "relevance" in tokens and paragraph:
                critic_data.append({
                    "instruction": CRITIC_PROMPTS["relevance_instruction"],
                    "input": CRITIC_PROMPTS["relevance_input"].format(
                        instruction=instruction, evidence=paragraph[:500]
                    ),
                    "output": tokens["relevance"],
                    "task": "relevance"
                })

            # 3. Groundedness 任务（需要有段落和支持度标签）
            if "support" in tokens and paragraph and clean_output:
                critic_data.append({
                    "instruction": CRITIC_PROMPTS["groundedness_instruction"],
                    "input": CRITIC_PROMPTS["groundedness_input"].format(
                        instruction=instruction,
                        output=clean_output[:300],
                        evidence=paragraph[:500]
                    ),
                    "output": tokens["support"],
                    "task": "groundedness"
                })

            # 4. Utility 任务
            if "utility" in tokens and clean_output:
                critic_data.append({
                    "instruction": CRITIC_PROMPTS["utility_instruction"],
                    "input": CRITIC_PROMPTS["utility_input"].format(
                        instruction=instruction, output=clean_output[:500]
                    ),
                    "output": tokens["utility"],
                    "task": "utility"
                })

    # 打乱并分割 train/dev
    random.shuffle(critic_data)

    # 统计
    task_counts = Counter(item["task"] for item in critic_data)
    label_counts = Counter(item["output"] for item in critic_data)
    print(f"  总样本数: {len(critic_data)}")
    print(f"  任务分布: {dict(task_counts)}")
    print(f"  标签分布: {dict(label_counts)}")

    # 截取并保存
    if len(critic_data) > max_samples:
        critic_data = critic_data[:max_samples]

    dev_size = min(1500, len(critic_data) // 10)
    train_data = critic_data[dev_size:]
    dev_data = critic_data[:dev_size]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    train_path = output_path.replace(".json", "_train.json")
    dev_path = output_path.replace(".json", "_dev.json")

    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    with open(dev_path, 'w', encoding='utf-8') as f:
        json.dump(dev_data, f, indent=2, ensure_ascii=False)

    print(f"  训练集: {train_path} ({len(train_data)} 条)")
    print(f"  验证集: {dev_path} ({len(dev_data)} 条)")
    print(f"  [OK] Critic 数据生成完成 ✅")
    return train_path


# ============================================================
#  评测数据下载与处理
# ============================================================

def download_eval_data(output_dir):
    """从 HuggingFace 下载并处理评测数据集"""
    print(f"\n[Eval] 下载并处理评测数据...")

    os.makedirs(output_dir, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        print("  [!] 需要安装 datasets: pip install datasets")
        return

    # ---- 1. PopQA ----
    print("\n  [1/4] PopQA (短答案 QA)...")
    try:
        popqa = load_dataset("akariasai/PopQA", split="test")
        popqa_output = []
        for item in popqa:
            entry = {
                "question": item.get("question", ""),
                "answers": item.get("possible_answers", []),
                "instruction": item.get("question", ""),
                "output": "",  # 空，由模型生成
                "topic": item.get("subj", ""),
                "s_pop": item.get("s_pop", 0),
            }
            # 预留 ctxs 字段，评测时由检索器填充
            entry["ctxs"] = []
            popqa_output.append(entry)

        popqa_path = os.path.join(output_dir, "popqa_longtail_w_gs.jsonl")
        with open(popqa_path, 'w', encoding='utf-8') as f:
            for item in popqa_output:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"    保存: {popqa_path} ({len(popqa_output)} 条)")
    except Exception as e:
        print(f"    [WARN] PopQA 下载失败: {e}")

    # ---- 2. ARC-Challenge ----
    print("\n  [2/4] ARC-Challenge (科学推理)...")
    try:
        arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        arc_output = []
        for item in arc:
            choices = item.get("choices", {})
            labels = choices.get("label", [])
            texts = choices.get("text", [])
            choices_text = " ".join([f"({l}) {t}" for l, t in zip(labels, texts)])
            answer_key = item.get("answerKey", "")

            entry = {
                "instruction": f"{item['question']}\n{choices_text}",
                "question": item["question"],
                "choices": choices_text,
                "answer": answer_key,
                "output": "",
                "ctxs": [],
            }
            arc_output.append(entry)

        arc_path = os.path.join(output_dir, "arc_challenge_processed.jsonl")
        with open(arc_path, 'w', encoding='utf-8') as f:
            for item in arc_output:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"    保存: {arc_path} ({len(arc_output)} 条)")
    except Exception as e:
        print(f"    [WARN] ARC-Challenge 下载失败: {e}")

    # ---- 3. PubHealth ----
    print("\n  [3/4] PubHealth (健康事实核查)...")
    try:
        # PubHealth 可能需要从不同来源下载
        try:
            pubhealth = load_dataset("bigbio/pubhealth", split="test", trust_remote_code=True)
        except Exception:
            pubhealth = load_dataset("health_fact", split="test", trust_remote_code=True)

        pubhealth_output = []
        for item in pubhealth:
            claim = item.get("claim", item.get("main_text", ""))
            label = item.get("label", item.get("label_text", ""))

            if not claim:
                continue

            entry = {
                "instruction": f"Is the following claim true or false? {claim}",
                "question": claim,
                "answer": str(label),
                "output": "",
                "ctxs": [],
            }
            pubhealth_output.append(entry)

        health_path = os.path.join(output_dir, "health_claims_processed.jsonl")
        with open(health_path, 'w', encoding='utf-8') as f:
            for item in pubhealth_output:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"    保存: {health_path} ({len(pubhealth_output)} 条)")
    except Exception as e:
        print(f"    [WARN] PubHealth 下载失败: {e}")
        print(f"    可手动下载后处理")

    # ---- 4. TriviaQA ----
    print("\n  [4/4] TriviaQA (开放域 QA)...")
    try:
        triviaqa = load_dataset("trivia_qa", "unfiltered.nocontext", split="validation[:2000]")
        triviaqa_output = []
        for item in triviaqa:
            question = item.get("question", "")
            answer = item.get("answer", {})
            aliases = answer.get("aliases", [])
            value = answer.get("value", "")

            entry = {
                "instruction": question,
                "question": question,
                "answers": aliases if aliases else [value],
                "output": "",
                "ctxs": [],
            }
            triviaqa_output.append(entry)

        triviaqa_path = os.path.join(output_dir, "triviaqa_test_w_gs.jsonl")
        with open(triviaqa_path, 'w', encoding='utf-8') as f:
            for item in triviaqa_output:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"    保存: {triviaqa_path} ({len(triviaqa_output)} 条)")
    except Exception as e:
        print(f"    [WARN] TriviaQA 下载失败: {e}")

    # ---- 总结 ----
    print(f"\n  [OK] 评测数据准备完成 ✅")
    print(f"  输出目录: {output_dir}")
    eval_files = [f for f in os.listdir(output_dir) if f.endswith(('.jsonl', '.json'))]
    for f in sorted(eval_files):
        size = os.path.getsize(os.path.join(output_dir, f))
        print(f"    {f}: {size/1024/1024:.1f} MB")


# ============================================================
#  主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Self-RAG 数据准备脚本")
    parser.add_argument("--task", type=str, default="all",
                        choices=["all", "critic", "eval"],
                        help="要执行的任务")
    parser.add_argument("--generator_data", type=str,
                        default="data/generator/train.jsonl",
                        help="Generator 训练数据路径")
    parser.add_argument("--critic_output", type=str,
                        default="data/critic/critic_train_data.json",
                        help="Critic 数据输出路径")
    parser.add_argument("--eval_output_dir", type=str,
                        default="data/eval",
                        help="评测数据输出目录")
    parser.add_argument("--max_critic_samples", type=int, default=50000,
                        help="Critic 数据最大样本数")
    args = parser.parse_args()

    print("=" * 60)
    print(" Self-RAG 数据准备")
    print("=" * 60)
    print(f"  任务: {args.task}")

    if args.task in ["all", "critic"]:
        if os.path.exists(args.generator_data):
            generate_critic_data(
                args.generator_data,
                args.critic_output,
                max_samples=args.max_critic_samples
            )
        else:
            print(f"\n  [!] Generator 数据不存在: {args.generator_data}")
            print(f"      请先运行: huggingface-cli download selfrag/selfrag_train_data --repo-type dataset --local-dir data/generator/")

    if args.task in ["all", "eval"]:
        download_eval_data(args.eval_output_dir)

    print(f"\n{'='*60}")
    print(f" 数据准备完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
