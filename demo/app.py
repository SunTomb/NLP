import gradio as gr
from vllm import LLM, SamplingParams
import re
import argparse
import os
import html

# 解析命令行参数获取模型路径
parser = argparse.ArgumentParser()
# 动态获取项目根目录，兼容不同路径下的执行
default_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "generator_llama2_7b")
parser.add_argument("--model_path", type=str, default=default_model_path, help="Path to the trained model")
args = parser.parse_args()

print(f"Loading model from {args.model_path}...")
try:
    llm = LLM(model=args.model_path, dtype="bfloat16", tensor_parallel_size=1, gpu_memory_utilization=0.8)
    # CRITICAL FIX: skip_special_tokens=False 必须设置，否则 vLLM 会在解码时抹除所有特殊标记 (如 [Retrieval])
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=512, skip_special_tokens=False)
except Exception as e:
    print(f"Failed to load vLLM: {e}. Running in dummy mode for UI preview.")
    llm = None
    sampling_params = None

TOKEN_STYLES = {
    "retrieval": "color: #1d4ed8; font-weight: 700; background-color: #dbeafe; padding: 2px 6px; border-radius: 6px; font-size: 0.85em; border: 1px solid #bfdbfe;",
    "no_retrieval": "color: #4b5563; font-weight: 700; background-color: #f3f4f6; padding: 2px 6px; border-radius: 6px; font-size: 0.85em; border: 1px solid #e5e7eb;",
    "relevant": "color: #15803d; font-weight: 700; background-color: #dcfce7; padding: 2px 6px; border-radius: 6px; font-size: 0.85em; border: 1px solid #bbf7d0;",
    "irrelevant": "color: #b91c1c; font-weight: 700; background-color: #fee2e2; padding: 2px 6px; border-radius: 6px; font-size: 0.85em; border: 1px solid #fecaca;",
    "partially": "color: #a16207; font-weight: 700; background-color: #fef08a; padding: 2px 6px; border-radius: 6px; font-size: 0.85em; border: 1px solid #fde047;",
    "utility": "color: #7e22ce; font-weight: 700; background-color: #f3e8ff; padding: 2px 6px; border-radius: 6px; font-size: 0.85em; border: 1px solid #e9d5ff;",
}

TOKEN_PATTERNS = [
    (r'\[No Retrieval\]', "no_retrieval"),
    (r'\[Retrieval\]', "retrieval"),
    (r'\[Relevant\]|\[Fully supported\]', "relevant"),
    (r'\[Irrelevant\]|\[No support / Contradictory\]', "irrelevant"),
    (r'\[Partially supported\]', "partially"),
    (r'\[Utility:\d\]', "utility"),
]

TOKEN_REGEX = re.compile("|".join(f"({pattern})" for pattern, _ in TOKEN_PATTERNS))


def token_span(token, style_name):
    return f'<span style="{TOKEN_STYLES[style_name]}">{token}</span>'


def format_tokens(text):
    text = html.escape(text)

    def replace_token(match):
        token = match.group(0)
        for pattern, style_name in TOKEN_PATTERNS:
            if re.fullmatch(pattern, token):
                return token_span(token, style_name)
        return token

    return TOKEN_REGEX.sub(replace_token, text)

def generate_answer(instruction):
    if not instruction.strip():
        return "<p style='color:red;'>⚠️ 提示：输入不能为空，请输入您的问题。</p>"
        
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    
    if llm is None:
        # Dummy response for UI test
        raw_text = "[No Retrieval] The author of Harry Potter is J.K. Rowling. [Utility:5]"
    else:
        try:
            outputs = llm.generate([prompt], sampling_params)
            raw_text = outputs[0].outputs[0].text
        except Exception as e:
            return f"<div style='color:red; padding:20px;'><b>生成失败 (可能提示词过长导致 OOM 或超长)：</b><br>{str(e)}</div>"
    
    html_output = format_tokens(raw_text)
    html_output = html_output.replace('\n', '<br>')
    
    # 包装在美观的容器中
    final_html = f"""
    <div style="background: white; border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); border: 1px solid #e5e7eb; font-family: system-ui, -apple-system, sans-serif; line-height: 1.6; color: #1f2937;">
        {html_output}
    </div>
    """
    return final_html

css = """
.header-banner {
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
    color: white;
    padding: 2rem;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
}
.header-banner h1 {
    margin: 0;
    font-size: 2.5rem;
    font-weight: 800;
    letter-spacing: -0.025em;
    color: white !important;
}
.header-banner p {
    margin-top: 0.5rem;
    font-size: 1.1rem;
    opacity: 0.9;
}
.legend-box {
    padding: 1rem;
    background: #f8fafc;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
}
.legend-item {
    display: inline-block;
    margin-bottom: 0.5rem;
}
/* Critical: Highlighting Tokens */
.token-retrieval { color: #1d4ed8 !important; font-weight: bold !important; background-color: #dbeafe !important; padding: 2px 6px !important; border-radius: 6px !important; font-size: 0.85em !important; border: 1px solid #bfdbfe !important; }
.token-no-retrieval { color: #4b5563 !important; font-weight: bold !important; background-color: #f3f4f6 !important; padding: 2px 6px !important; border-radius: 6px !important; font-size: 0.85em !important; border: 1px solid #e5e7eb !important; }
.token-relevant { color: #15803d !important; font-weight: bold !important; background-color: #dcfce7 !important; padding: 2px 6px !important; border-radius: 6px !important; font-size: 0.85em !important; border: 1px solid #bbf7d0 !important; }
.token-irrelevant { color: #b91c1c !important; font-weight: bold !important; background-color: #fee2e2 !important; padding: 2px 6px !important; border-radius: 6px !important; font-size: 0.85em !important; border: 1px solid #fecaca !important; }
.token-partially { color: #a16207 !important; font-weight: bold !important; background-color: #fef08a !important; padding: 2px 6px !important; border-radius: 6px !important; font-size: 0.85em !important; border: 1px solid #fde047 !important; }
.token-utility { color: #7e22ce !important; font-weight: bold !important; background-color: #f3e8ff !important; padding: 2px 6px !important; border-radius: 6px !important; font-size: 0.85em !important; border: 1px solid #e9d5ff !important; }
"""

with gr.Blocks(title="Self-RAG 交互演示") as demo:
    # 顶部 Banner
    gr.HTML("""
    <div class="header-banner">
        <h1>🔍 Self-RAG 交互式系统</h1>
        <p>探索大模型如何通过「反思 Token」进行自我批判与自适应检索</p>
    </div>
    """)
    
    with gr.Row():
        # 左侧：控制与输入面板
        with gr.Column(scale=1):
            gr.Markdown("### 📝 使用说明\n在输入框中输入您的问题。由于当前运行于 **No Retrieval** 模式（即未挂接外部维基百科等检索库），因此您可以着重观察：\n\n1. 模型在遇到何种问题时会主动发出 `[Retrieval]` 信号。\n2. 在纯依赖内部知识作答后，模型对其回答给出的 `[Utility]` 评分。\n\n*⚠️ 提示：Self-RAG 基于指令微调，主要针对单论短回答优化。当您输入过长或包含多个问题时，它倾向于只对首个问题作答。*")
            
            instruction_input = gr.Textbox(
                lines=5, 
                placeholder="例如: Who is the author of Harry Potter? ...", 
                label="🧑‍💻 用户指令 (Instruction)",
                elem_classes=["input-box"]
            )
            
            submit_btn = gr.Button("🚀 立即生成", variant="primary", size="lg")
            
            gr.Markdown("---")
            gr.Markdown("### 💡 快捷测试用例")
            gr.Examples(
                examples=[
                    "Who is the author of Harry Potter?",
                    "Can you explain what quantum computing is in simple terms?",
                    "What are the major differences between Llama-2 and Qwen2.5?",
                    "Write a short python function to calculate Fibonacci numbers."
                ],
                inputs=instruction_input,
                label=""
            )
            
        # 右侧：输出与图例说明
        with gr.Column(scale=1):
            gr.Markdown("### 🤖 模型响应解析\n生成的文本将嵌入模型自身的思维过程：")
            
            # 图例
            gr.HTML(f"""
            <div style="padding: 1rem; background: #f8fafc; border-radius: 8px; border: 1px solid #e2e8f0;">
                <div style="margin-bottom: 8px; font-weight: 600; color: #475569;">反思 Token 图例：</div>
                <div style="display: inline-block; margin-bottom: 0.5rem;">{token_span('[Retrieval]', 'retrieval')} : 模型认为需要检索外部知识</div><br>
                <div style="display: inline-block; margin-bottom: 0.5rem;">{token_span('[No Retrieval]', 'no_retrieval')} : 无需检索，可直接作答</div><br>
                <div style="display: inline-block; margin-bottom: 0.5rem;">{token_span('[Relevant]', 'relevant')} / {token_span('[Fully supported]', 'relevant')} : 高质量、强支持度的内容</div><br>
                <div style="display: inline-block; margin-bottom: 0.5rem;">{token_span('[Utility:1]', 'utility')}~{token_span('[Utility:5]', 'utility')} : 最终对用户指令的有效性总评</div>
            </div>
            """)
            
            gr.Markdown("<br>")
            output_html = gr.HTML(label="输出结果", value='<div style="padding: 2rem; text-align: center; color: #94a3b8; border: 2px dashed #cbd5e1; border-radius: 12px;">生成的文本及反思过程将在此处显示</div>')

    # 绑定事件
    submit_btn.click(
        fn=lambda: '<div style="padding: 2rem; text-align: center; color: #3b82f6;"><i class="fas fa-spinner fa-spin"></i> 正在深思熟虑中...</div>', 
        inputs=None, 
        outputs=output_html,
        queue=False
    ).then(
        fn=generate_answer, 
        inputs=instruction_input, 
        outputs=output_html,
        api_name=False
    )

if __name__ == "__main__":
    print("Launching Gradio demo...", flush=True)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, css=css)
