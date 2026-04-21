#!/bin/bash
# ============================================================
# Self-RAG 项目环境部署脚本
# 在集群上执行: bash scripts/setup_env.sh
# ============================================================
set -e

echo "==========================================="
echo " Self-RAG 环境部署"
echo "==========================================="

# ---- 0. 项目根目录（请修改为你的实际路径） ----
export PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
echo "[INFO] 项目目录: $PROJECT_DIR"

# ---- 1. 创建目录结构 ----
echo "[Step 1] 创建项目目录结构..."
mkdir -p "$PROJECT_DIR"/{models,data/{critic,generator,eval,chinese/{eval,wiki_zh_index}},outputs,results,figures/demo,scripts/{eval,ablation},demo,report,logs}
echo "[OK] 目录结构已创建"

# ---- 2. 检查系统环境 ----
echo ""
echo "[Step 2] 检查系统环境..."
echo "--- GPU 信息 ---"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "[WARN] nvidia-smi 不可用，可能需要在 GPU 节点上运行"
echo ""
echo "--- CUDA 版本 ---"
nvcc --version 2>/dev/null || echo "[WARN] nvcc 不可用，检查 module 系统"
echo ""
echo "--- Conda 版本 ---"
conda --version 2>/dev/null || echo "[WARN] conda 不可用，需要安装 miniconda"
echo ""

# ---- 3. 创建 Conda 环境 ----
echo "[Step 3] 创建 Conda 环境 'selfrag'..."
if conda env list | grep -q "selfrag"; then
    echo "[INFO] 环境 'selfrag' 已存在，跳过创建"
else
    conda create -n selfrag python=3.10 -y
    echo "[OK] Conda 环境已创建"
fi

# ---- 4. 激活环境并安装依赖 ----
echo "[Step 4] 安装依赖..."
eval "$(conda shell.bash hook)"
conda activate selfrag

# 4.1 PyTorch（CUDA 12.1）
echo "  安装 PyTorch..."
pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4.2 核心 NLP 依赖
echo "  安装 Transformers 和相关依赖..."
pip install transformers==4.44.0 tokenizers accelerate==0.33.0
pip install deepspeed==0.14.4
pip install datasets evaluate sentencepiece protobuf
pip install peft bitsandbytes

# 4.3 vllm（推理引擎）
echo "  安装 vllm..."
pip install vllm==0.5.5

# 4.4 训练与日志
echo "  安装训练工具..."
pip install tensorboard jsonlines tqdm scikit-learn nltk spacy einops

# 4.5 检索相关
echo "  安装检索依赖..."
pip install faiss-gpu sentence-transformers

# 4.6 可视化与 Demo
echo "  安装可视化和 Demo 依赖..."
pip install matplotlib seaborn gradio

# 4.7 评测
echo "  安装评测工具..."
pip install sacrebleu rouge-score

# 4.8 Flash Attention（可选，编译可能较慢）
echo "  尝试安装 flash-attn（可能需要几分钟）..."
pip install flash-attn --no-build-isolation 2>/dev/null || echo "[WARN] flash-attn 安装失败，训练时去掉 --use_flash_attn 参数即可"

# 4.9 Google Drive 下载工具
pip install gdown

# ---- 5. 验证安装 ----
echo ""
echo "[Step 5] 验证安装..."
python -c "
import torch
print(f'  PyTorch:       {torch.__version__}')
print(f'  CUDA 可用:     {torch.cuda.is_available()}')
print(f'  GPU 数量:      {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}:         {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem / 1e9:.1f} GB)')
"

python -c "from vllm import LLM; print('  vllm:          OK')" 2>/dev/null || echo "  vllm:          FAILED (可能需要在 GPU 节点验证)"
python -c "import deepspeed; print(f'  DeepSpeed:     {deepspeed.__version__}')"
python -c "import transformers; print(f'  Transformers:  {transformers.__version__}')"
python -c "import datasets; print(f'  Datasets:      {datasets.__version__}')"

echo ""
echo "==========================================="
echo " 环境部署完成！"
echo "==========================================="
echo ""
echo "后续步骤:"
echo "  1. bash scripts/download_models.sh  (下载模型)"
echo "  2. bash scripts/download_data.sh    (下载数据)"
echo "  3. python scripts/quick_inference.py (快速推理验证)"
