import gradio as gr
from vllm import LLM, SamplingParams
import re
import argparse

# 解析命令行参数获取模型路径
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="../outputs/generator_llama2_7b", help="Path to the trained model")
args = parser.parse_args()

print(f"Loading model from {args.model_path}...")
llm = LLM(model=args.model_path, dtype="bfloat16", tensor_parallel_size=1, gpu_memory_utilization=0.8)
sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=200)

def format_tokens(text):
    """为反思 Token 添加 HTML 颜色高亮"""
    # 检索决策 (蓝色)
    text = re.sub(r'(\[Retrieval\])', r'<span style="color: #2563eb; font-weight: bold; background-color: #dbeafe; padding: 2px 4px; border-radius: 4px;">\1</span>', text)
    text = re.sub(r'(\[No Retrieval\])', r'<span style="color: #4b5563; font-weight: bold; background-color: #f3f4f6; padding: 2px 4px; border-radius: 4px;">\1</span>', text)
    
    # 质量评估 (绿色/红色)
    text = re.sub(r'(\[Relevant\]|\[Fully supported\])', r'<span style="color: #16a34a; font-weight: bold; background-color: #dcfce7; padding: 2px 4px; border-radius: 4px;">\1</span>', text)
    text = re.sub(r'(\[Irrelevant\]|\[No support / Contradictory\])', r'<span style="color: #dc2626; font-weight: bold; background-color: #fee2e2; padding: 2px 4px; border-radius: 4px;">\1</span>', text)
    text = re.sub(r'(\[Partially supported\])', r'<span style="color: #ca8a04; font-weight: bold; background-color: #fef08a; padding: 2px 4px; border-radius: 4px;">\1</span>', text)
    
    # 效用 (紫色)
    text = re.sub(r'(\[Utility:\d\])', r'<span style="color: #9333ea; font-weight: bold; background-color: #f3e8ff; padding: 2px 4px; border-radius: 4px;">\1</span>', text)
    
    return text

def generate_answer(instruction):
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    outputs = llm.generate([prompt], sampling_params)
    raw_text = outputs[0].outputs[0].text
    
    html_output = format_tokens(raw_text)
    
    # 简单的格式化，防止换行失效
    html_output = html_output.replace('\n', '<br>')
    
    return html_output

with gr.Blocks() as demo:
    gr.Markdown("# Self-RAG 交互式演示 (No Retrieval Mode)")
    gr.Markdown("当前演示仅调用 Generator 模型，未挂载外部检索器。你可以观察模型在仅靠内部参数记忆时，是否会生成 `[Retrieval]` 请求，以及最终对生成内容的自我反思 (如 `[Utility]`)。")
    
    with gr.Row():
        with gr.Column():
            instruction_input = gr.Textbox(lines=4, placeholder="Ask a question...", label="Instruction")
            submit_btn = gr.Button("Generate", variant="primary")
            gr.Examples(
                examples=[
                    "Who is the author of Harry Potter?",
                    "What is the capital of France?",
                    "Can you explain what quantum computing is in simple terms?"
                ],
                inputs=instruction_input
            )
        
        with gr.Column():
            output_html = gr.HTML(label="Self-RAG Response with Reflection Tokens")

    submit_btn.click(fn=generate_answer, inputs=instruction_input, outputs=output_html)

if __name__ == "__main__":
    print("Launching Gradio demo...", flush=True)
    # Gradio 6.0: share=True 生成公网链接
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

