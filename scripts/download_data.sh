#!/bin/bash
# ============================================================
# Self-RAG 数据下载脚本
# 在集群上执行: bash scripts/download_data.sh
# ============================================================
set -e

export PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

# HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

echo "==========================================="
echo " Self-RAG 数据下载"
echo "==========================================="

# ---- 1. Critic 训练数据（GPT-4 标注，Google Drive）----
echo ""
echo "[1/3] 下载 Critic 训练数据..."
if [ -f "data/critic/critic_train_data.json" ]; then
    echo "  [跳过] 已存在"
else
    echo "  尝试使用 gdown 从 Google Drive 下载..."
    gdown 1IN1XcIOYtRIGWITJ4LKRgfITT-uUwk_W -O data/critic/critic_train_data.json 2>/dev/null || {
        echo "  [WARN] gdown 下载失败"
        echo "  请手动下载: https://drive.google.com/file/d/1IN1XcIOYtRIGWITJ4LKRgfITT-uUwk_W/view"
        echo "  然后放到: $PROJECT_DIR/data/critic/critic_train_data.json"
    }
fi

# ---- 2. Generator 训练数据（150K 实例，HuggingFace）----
echo ""
echo "[2/3] 下载 Generator 训练数据 (150K)..."
if [ -d "data/generator" ] && ls data/generator/*.jsonl 1>/dev/null 2>&1; then
    echo "  [跳过] 已存在"
else
    echo "  从 HuggingFace 下载..."
    huggingface-cli download selfrag/selfrag_train_data \
        --local-dir data/generator/ \
        --repo-type dataset \
        --local-dir-use-symlinks False 2>/dev/null || {
        echo "  [WARN] HuggingFace 下载失败"
        echo "  备选方案: gdown 从 Google Drive 下载"
        gdown 10G_FozUV4u27EX0NjwVe-3YMUMeTwuLk -O data/generator/selfrag_train_data.jsonl 2>/dev/null || {
            echo "  请手动下载: https://drive.google.com/file/d/10G_FozUV4u27EX0NjwVe-3YMUMeTwuLk/view"
            echo "  然后放到: $PROJECT_DIR/data/generator/"
        }
    }
fi

# ---- 3. 评测数据集 ----
echo ""
echo "[3/3] 下载评测数据集..."
if [ -d "data/eval/eval_data" ] || ls data/eval/*.jsonl 1>/dev/null 2>&1; then
    echo "  [跳过] 已存在"
else
    echo "  尝试使用 gdown 从 Google Drive 下载..."
    gdown 1TLKhWjez63H4uBtgCxyoyJsZi-IMgnDb -O data/eval/eval_data.zip 2>/dev/null && {
        cd data/eval
        unzip -o eval_data.zip
        rm -f eval_data.zip
        cd "$PROJECT_DIR"
        echo "  [OK] 评测数据解压完成"
    } || {
        echo "  [WARN] gdown 下载失败"
        echo "  请手动下载: https://drive.google.com/file/d/1TLKhWjez63H4uBtgCxyoyJsZi-IMgnDb/view"
        echo "  然后解压到: $PROJECT_DIR/data/eval/"
    }
fi

# ---- 4. 下载 Contriever 检索模型（用于检索 Demo）----
echo ""
echo "[bonus] 下载检索 demo 语料..."
cd "$PROJECT_DIR/self-rag/retrieval_lm"
if [ -d "enwiki_2020_intro_only" ]; then
    echo "  [跳过] Demo 语料已存在"
else
    bash download_demo_corpus.sh 2>/dev/null || echo "  [WARN] Demo 语料下载失败，不影响主实验"
fi
cd "$PROJECT_DIR"

# ---- 5. 数据概览 ----
echo ""
echo "==========================================="
echo " 数据概览"
echo "==========================================="

python -c "
import os, json, glob

# Critic 数据
critic_file = 'data/critic/critic_train_data.json'
if os.path.exists(critic_file):
    data = json.load(open(critic_file))
    if isinstance(data, list):
        print(f'  Critic 训练数据:   {len(data):,} 条')
        if data:
            print(f'    字段: {list(data[0].keys())}')
    elif isinstance(data, dict):
        print(f'  Critic 训练数据:   dict with keys {list(data.keys())}')
else:
    print('  Critic 训练数据:   未下载')

# Generator 数据
gen_files = glob.glob('data/generator/**/*.jsonl', recursive=True) + glob.glob('data/generator/**/*.json', recursive=True)
if gen_files:
    total = 0
    for f in gen_files:
        with open(f) as fh:
            if f.endswith('.jsonl'):
                count = sum(1 for _ in fh)
            else:
                data = json.load(fh)
                count = len(data) if isinstance(data, list) else 1
        total += count
        print(f'  Generator 数据文件: {os.path.basename(f)} ({count:,} 条)')
    print(f'  Generator 数据总量: {total:,} 条')
else:
    print('  Generator 训练数据: 未下载')

# 评测数据
eval_files = glob.glob('data/eval/**/*.jsonl', recursive=True) + glob.glob('data/eval/**/*.json', recursive=True)
if eval_files:
    print(f'  评测数据文件数:   {len(eval_files)}')
    for f in sorted(eval_files)[:10]:
        with open(f) as fh:
            if f.endswith('.jsonl'):
                count = sum(1 for _ in fh)
            else:
                data = json.load(fh)
                count = len(data) if isinstance(data, list) else 1
        print(f'    {os.path.basename(f):45s} {count:>6,} 条')
else:
    print('  评测数据:          未下载')
"

echo ""
echo "数据下载完成！"
echo "后续步骤: python scripts/quick_inference.py"
