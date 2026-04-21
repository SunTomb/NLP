"""
Self-RAG 交互式 Demo 系统
使用 Gradio 构建 Web UI，展示 Self-RAG 的完整推理过程

在集群上执行：
  conda activate selfrag
  python demo/app.py --model_path outputs/generator_llama2_7b

然后通过 SSH 端口转发在本地浏览器访问：
  ssh -L 7860:localhost:7860 username@cluster_ip
  浏览器打开: http://localhost:7860
"""

import argparse
import json
import re
import gradio as gr


def load_model(model_path, gpu_memory_utilization=0.85):
    """加载 Self-RAG 模型"""
    from vllm import LLM, SamplingParams
    print(f"[INFO] 加载模型: {model_path}")
    model = LLM(
        model_path,
        dtype="half",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=2048,
    )
    print("[OK] 模型加载完成")
    return model


def format_prompt(query, paragraph=None):
    prompt = f"### Instruction:\n{query}\n\n### Response:\n"
    if paragraph:
        prompt += f"[Retrieval]<paragraph>{paragraph}</paragraph>"
    return prompt


def parse_reflection_tokens(text):
    """解析输出中的反思 Token"""
    tokens = {}

    # 检索 Token
    if "[No Retrieval]" in text:
        tokens["检索决策"] = "🔵 不需要检索 [No Retrieval]"
    elif "[Retrieval]" in text:
        tokens["检索决策"] = "🟢 需要检索 [Retrieval]"

    # 相关性 Token
    if "[Relevant]" in text:
        tokens["文档相关性"] = "✅ 相关 [Relevant]"
    elif "[Irrelevant]" in text:
        tokens["文档相关性"] = "❌ 不相关 [Irrelevant]"

    # 支持度 Token
    if "[Fully supported]" in text:
        tokens["证据支持度"] = "🟢 完全支持 [Fully supported]"
    elif "[Partially supported]" in text:
        tokens["证据支持度"] = "🟡 部分支持 [Partially supported]"
    elif "[No support / Contradictory]" in text:
        tokens["证据支持度"] = "🔴 不支持/矛盾 [No support / Contradictory]"

    # 有用性 Token
    for i in range(5, 0, -1):
        if f"[Utility:{i}]" in text:
            stars = "⭐" * i
            tokens["回答有用性"] = f"{stars} [Utility:{i}]"
            break

    return tokens


def clean_output(text):
    """清理输出，去除反思 Token"""
    patterns = [
        r"\[No Retrieval\]", r"\[Retrieval\]", r"\[Continue to Use Evidence\]",
        r"\[Relevant\]", r"\[Irrelevant\]",
        r"\[Fully supported\]", r"\[Partially supported\]", r"\[No support / Contradictory\]",
        r"\[Utility:[1-5]\]",
        r"<paragraph>.*?</paragraph>",
        r"</s>",
    ]
    result = text
    for p in patterns:
        result = re.sub(p, "", result)
    return result.strip()


def create_demo(model):
    """创建 Gradio Demo UI"""
    from vllm import SamplingParams

    def generate_response(query, paragraph, temperature, max_tokens, top_k):
        if not query.strip():
            return "请输入问题", "", ""

        # 生成 Self-RAG 输出
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=1.0,
            max_tokens=int(max_tokens),
            skip_special_tokens=False,
        )

        prompt = format_prompt(query, paragraph if paragraph.strip() else None)
        preds = model.generate([prompt], sampling_params)
        raw_output = preds[0].outputs[0].text

        # 解析反思 Token
        tokens = parse_reflection_tokens(raw_output)
        clean = clean_output(raw_output)

        # 格式化反思分析
        analysis = "## 🔍 反思 Token 分析\n\n"
        if tokens:
            for key, val in tokens.items():
                analysis += f"**{key}**: {val}\n\n"
        else:
            analysis += "未检测到反思 Token\n"

        analysis += "\n---\n## 📝 原始输出（含反思 Token）\n\n"
        analysis += f"```\n{raw_output}\n```"

        return clean, analysis, raw_output

    # 示例输入
    examples = [
        ["What is 2+2?", "", 0.0, 200, 10],
        ["Can you tell me the difference between llamas and alpacas?", "", 0.0, 200, 10],
        ["Can you tell me the difference between llamas and alpacas?",
         "The alpaca (Lama pacos) is a species of South American camelid mammal. It is similar to, and often confused with, the llama. Alpacas are considerably smaller than llamas, and unlike llamas, they were not bred to be working animals, but were bred specifically for their fiber.",
         0.0, 200, 10],
        ["What is overfitting in machine learning?",
         "In statistics, overfitting is the production of an analysis that corresponds too closely or exactly to a particular set of data, and may therefore fail to fit additional data or predict future observations reliably.",
         0.0, 200, 10],
        ["中国的首都是哪里？", "", 0.0, 200, 10],
        ["Is it true that drinking 8 glasses of water a day is necessary for good health?", "", 0.0, 200, 10],
    ]

    with gr.Blocks(
        title="Self-RAG Demo",
        theme=gr.themes.Soft(primary_hue="blue"),
    ) as demo:
        gr.Markdown(
            """
            # 🔬 Self-RAG: 自适应检索增强生成系统
            
            > **Self-RAG** 通过反思 Token 实现自主检索决策和生成质量自评。
            > 模型会判断是否需要检索（`[Retrieval]`/`[No Retrieval]`），
            > 评估检索文档的相关性（`[Relevant]`/`[Irrelevant]`），
            > 以及生成内容的支持度和有用性。
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                query_input = gr.Textbox(
                    label="📋 输入问题",
                    placeholder="输入你的问题...",
                    lines=2,
                )
                paragraph_input = gr.Textbox(
                    label="📄 检索段落（可选）",
                    placeholder="如果有检索到的段落，粘贴在这里...",
                    lines=4,
                )
                with gr.Row():
                    temperature = gr.Slider(0, 1, value=0, step=0.1, label="Temperature")
                    max_tokens = gr.Slider(50, 500, value=200, step=50, label="Max Tokens")
                    top_k = gr.Slider(1, 20, value=10, step=1, label="Top-K Docs")

                submit_btn = gr.Button("🚀 生成回答", variant="primary")

            with gr.Column(scale=3):
                answer_output = gr.Textbox(label="💬 回答", lines=4)
                analysis_output = gr.Markdown(label="🔍 反思分析")
                raw_output = gr.Textbox(label="📝 原始输出", lines=3, visible=False)

        submit_btn.click(
            generate_response,
            inputs=[query_input, paragraph_input, temperature, max_tokens, top_k],
            outputs=[answer_output, analysis_output, raw_output],
        )

        gr.Examples(
            examples=examples,
            inputs=[query_input, paragraph_input, temperature, max_tokens, top_k],
        )

        gr.Markdown(
            """
            ---
            **NLP 课程大作业 | Self-RAG (14.4) | USTC 2026 Spring**
            """
        )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="outputs/generator_llama2_7b")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="生成公网链接")
    parser.add_argument("--gpu_memory", type=float, default=0.85)
    args = parser.parse_args()

    model = load_model(args.model_path, args.gpu_memory)
    demo = create_demo(model)
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
