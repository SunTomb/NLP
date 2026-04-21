#!/bin/bash
# Self-RAG 项目环境激活脚本
# 用法: source activate.sh

cd /NAS/yesh/NLP
eval "$(/NAS/yesh/miniconda3/bin/conda shell.bash hook)"
conda activate /NAS/yesh/NLP/.conda/selfrag

# 确保 conda 环境的 bin 在 PATH 最前面
export PATH="/NAS/yesh/NLP/.conda/selfrag/bin:$PATH"

export HF_CACHE=/NAS/yesh/hf_cache/hub
export HF_HOME=/NAS/yesh/hf_cache
export HUGGINGFACE_HUB_CACHE=/NAS/yesh/hf_cache/hub
export HF_ENDPOINT=https://hf-mirror.com
export PROJECT_DIR=/NAS/yesh/NLP
export PYTHONPATH=/NAS/yesh/NLP
echo "Self-RAG environment ready ✅"
echo "  项目目录: $PROJECT_DIR"
echo "  HF 缓存:  $HF_CACHE"
echo "  Python:   $(python --version) @ $(which python)"
