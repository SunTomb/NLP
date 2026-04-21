#!/bin/bash
# ============================================================
# Self-RAG 模型下载脚本
# 在集群上执行: bash scripts/download_models.sh
# ============================================================
set -e

export PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

# HuggingFace 镜像（国内加速，如不需要可注释掉）
export HF_ENDPOINT=https://hf-mirror.com

echo "==========================================="
echo " Self-RAG 模型下载"
echo "==========================================="
echo " 项目目录: $PROJECT_DIR"
echo " HF 镜像:  $HF_ENDPOINT"
echo "==========================================="

# ---- 1. 下载官方预训练 Self-RAG 7B（推理验证，~14GB）----
echo ""
echo "[1/4] 下载 selfrag/selfrag_llama2_7b (~14GB)..."
if [ -d "models/selfrag_llama2_7b" ] && [ -f "models/selfrag_llama2_7b/config.json" ]; then
    echo "  [跳过] 已存在"
else
    huggingface-cli download selfrag/selfrag_llama2_7b \
        --local-dir models/selfrag_llama2_7b \
        --local-dir-use-symlinks False
    echo "  [OK] 下载完成"
fi

# ---- 2. 下载 Llama 2 7B 基座模型（复现训练，~14GB）----
echo ""
echo "[2/4] 下载 meta-llama/Llama-2-7b-hf (~14GB)..."
echo "  [注意] 需要 HuggingFace 登录并接受 Meta Llama 2 使用协议"
echo "  如未登录，请先执行: huggingface-cli login"
if [ -d "models/Llama-2-7b-hf" ] && [ -f "models/Llama-2-7b-hf/config.json" ]; then
    echo "  [跳过] 已存在"
else
    huggingface-cli download meta-llama/Llama-2-7b-hf \
        --local-dir models/Llama-2-7b-hf \
        --local-dir-use-symlinks False || \
    echo "  [WARN] 下载失败，可能需要先: huggingface-cli login 并接受 Llama 2 协议"
fi

# ---- 3. 下载 Llama 3.1 8B（改进实验 I1，~16GB，可稍后下载）----
echo ""
echo "[3/4] 下载 meta-llama/Llama-3.1-8B (~16GB)..."
echo "  [注意] 改进实验用，非必需，可稍后下载"
read -p "  是否现在下载？[y/N]: " download_llama3
if [ "$download_llama3" = "y" ] || [ "$download_llama3" = "Y" ]; then
    if [ -d "models/Llama-3.1-8B" ] && [ -f "models/Llama-3.1-8B/config.json" ]; then
        echo "  [跳过] 已存在"
    else
        huggingface-cli download meta-llama/Llama-3.1-8B \
            --local-dir models/Llama-3.1-8B \
            --local-dir-use-symlinks False || \
        echo "  [WARN] 下载失败"
    fi
else
    echo "  [跳过]"
fi

# ---- 4. 下载 Qwen 2.5 7B（中文适配，~14GB，可稍后下载）----
echo ""
echo "[4/4] 下载 Qwen/Qwen2.5-7B (~14GB)..."
echo "  [注意] 中文适配实验用，非必需，可稍后下载"
read -p "  是否现在下载？[y/N]: " download_qwen
if [ "$download_qwen" = "y" ] || [ "$download_qwen" = "Y" ]; then
    if [ -d "models/Qwen2.5-7B" ] && [ -f "models/Qwen2.5-7B/config.json" ]; then
        echo "  [跳过] 已存在"
    else
        huggingface-cli download Qwen/Qwen2.5-7B \
            --local-dir models/Qwen2.5-7B \
            --local-dir-use-symlinks False || \
        echo "  [WARN] 下载失败"
    fi
else
    echo "  [跳过]"
fi

# ---- 5. 验证模型 ----
echo ""
echo "==========================================="
echo " 验证已下载模型"
echo "==========================================="
python -c "
import os
models_dir = 'models'
for name in sorted(os.listdir(models_dir)):
    model_dir = os.path.join(models_dir, name)
    if os.path.isdir(model_dir):
        config = os.path.join(model_dir, 'config.json')
        if os.path.exists(config):
            import json
            with open(config) as f:
                cfg = json.load(f)
            arch = cfg.get('architectures', ['Unknown'])[0]
            size = sum(os.path.getsize(os.path.join(model_dir, f)) for f in os.listdir(model_dir) if f.endswith(('.bin', '.safetensors')))
            print(f'  ✓ {name:30s} | {arch:30s} | {size/1e9:.1f} GB')
        else:
            print(f'  ✗ {name:30s} | config.json 缺失')
"

echo ""
echo "模型下载完成！"
echo "后续步骤: bash scripts/download_data.sh"
