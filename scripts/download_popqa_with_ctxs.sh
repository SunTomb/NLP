#!/bin/bash
# =============================================================================
# 下载包含 ctxs（检索段落）的 PopQA 评测数据
# =============================================================================

set -eo pipefail

cd "$(dirname "$0")/.."
source activate.sh 2>/dev/null || true

DATA_DIR="data/eval"
OUTPUT_FILE="${DATA_DIR}/popqa_longtail_w_gs.jsonl"
BACKUP_FILE="${DATA_DIR}/popqa_longtail_w_gs.jsonl.bak_no_ctxs"

mkdir -p "${DATA_DIR}"

echo "============================================================"
echo "  下载 PopQA 评测数据 (含检索段落 ctxs)"
echo "============================================================"

# ---- Step 1: 检查当前文件是否已包含非空 ctxs ----
echo "[1] 检查当前数据是否包含非空 ctxs 字段..."
HAS_CTXS=$(python -c "
import json
with open('${OUTPUT_FILE}', encoding='utf-8') as f:
    first = json.loads(f.readline())
has = 'ctxs' in first and len(first['ctxs']) > 0
print('yes' if has else 'no')
" 2>/dev/null || echo "no")

if [ "$HAS_CTXS" == "yes" ]; then
    echo "    ✅ 当前数据已包含非空 ctxs 字段，无需重新下载"
    python -c "
import json
with open('${OUTPUT_FILE}', encoding='utf-8') as f:
    first = json.loads(f.readline())
print(f'    ctxs 数量: {len(first[\"ctxs\"])}')
print(f'    ctxs[0] keys: {list(first[\"ctxs\"][0].keys())}')
print(f'    ctxs[0] title: {first[\"ctxs\"][0].get(\"title\", \"N/A\")[:60]}')
"
    exit 0
fi

echo "    ⚠️  当前数据不包含有效的 ctxs 字段（不存在或为空列表），需要下载..."

# ---- Step 2: 备份当前版本 ----
if [ -f "${OUTPUT_FILE}" ]; then
    echo "[2] 备份当前数据文件..."
    cp "${OUTPUT_FILE}" "${BACKUP_FILE}"
    echo "    备份到: ${BACKUP_FILE}"
fi

# ---- Step 3: 从 HuggingFace 下载 ----
echo "[3] 尝试从 HuggingFace 下载完整评测数据..."

python << 'PYEOF'
import os, json, sys

OUTPUT_FILE = "data/eval/popqa_longtail_w_gs.jsonl"

def check_ctxs(filepath):
    """检查文件第一行是否包含非空 ctxs"""
    with open(filepath, encoding='utf-8') as f:
        first = json.loads(f.readline())
    ctxs = first.get('ctxs', [])
    return len(ctxs) > 0, len(ctxs)

# ========== 方法 1: selfrag 官方 HF 数据集 ==========
print("  方法 1: 尝试从 selfrag/selfrag_train_data 下载...")
try:
    from huggingface_hub import hf_hub_download, list_repo_tree

    # 先列出仓库中的文件，找到 eval_data 目录
    print("    列出仓库文件结构...")
    entries = list(list_repo_tree("selfrag/selfrag_train_data", repo_type="dataset"))
    eval_files = [e.path for e in entries if 'popqa' in e.path.lower()]
    print(f"    找到 PopQA 相关文件: {eval_files}")

    # 下载最可能的文件
    candidates = [
        'eval_data/popqa_longtail_w_gs.jsonl',
        'popqa_longtail_w_gs.jsonl',
    ] + eval_files

    for fname in candidates:
        if not fname:
            continue
        try:
            print(f"    尝试下载: {fname}")
            filepath = hf_hub_download(
                repo_id='selfrag/selfrag_train_data',
                filename=fname,
                repo_type='dataset',
                cache_dir=os.environ.get('HF_HOME', '.cache'),
            )
            has, count = check_ctxs(filepath)
            if has:
                print(f"    ✅ 成功! 包含 {count} 个检索段落")
                import shutil
                shutil.copy2(filepath, OUTPUT_FILE)
                print(f"    已复制到: {OUTPUT_FILE}")
                sys.exit(0)
            else:
                print(f"    ⚠️  ctxs 为空 (len={count})")
        except Exception as e:
            print(f"    跳过 {fname}: {e}")

except Exception as e:
    print(f"    方法 1 失败: {e}")

# ========== 方法 2: 第三方 HF 仓库 ==========
print("\n  方法 2: 尝试从 awinml/popqa_longtail_w_gs 下载...")
try:
    from huggingface_hub import hf_hub_download, list_repo_tree
    entries = list(list_repo_tree("awinml/popqa_longtail_w_gs", repo_type="dataset"))
    all_files = [e.path for e in entries if e.path.endswith(('.jsonl', '.json', '.parquet'))]
    print(f"    仓库文件: {all_files}")

    for fname in all_files:
        try:
            filepath = hf_hub_download(
                repo_id='awinml/popqa_longtail_w_gs',
                filename=fname,
                repo_type='dataset',
                cache_dir=os.environ.get('HF_HOME', '.cache'),
            )
            has, count = check_ctxs(filepath)
            if has:
                print(f"    ✅ {fname}: 包含 {count} 个段落")
                import shutil
                shutil.copy2(filepath, OUTPUT_FILE)
                sys.exit(0)
            else:
                print(f"    ⚠️  {fname}: ctxs 为空或不存在")
        except Exception as e:
            print(f"    跳过 {fname}: {e}")
except Exception as e:
    print(f"    方法 2 失败: {e}")

# ========== 方法 3: 使用 Contriever 自行检索 ==========
print("\n" + "=" * 60)
print("  ❌ HuggingFace 上的数据均不包含非空 ctxs")
print("  需要使用 Contriever 自行检索段落")
print("=" * 60)
print()
print("  请运行以下步骤为 PopQA 数据添加检索段落：")
print()
print("  Step 1: 下载 Wikipedia 段落数据 (~2GB)")
print("    cd self-rag/retrieval_lm")
print("    wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz")
print("    gunzip psgs_w100.tsv.gz")
print()
print("  Step 2: 下载 Contriever-MSMARCO 嵌入 (~18GB)")
print("    wget https://dl.fbaipublicfiles.com/contriever/embeddings/contriever-msmarco/wikipedia_embeddings.tar")
print("    tar xvf wikipedia_embeddings.tar")
print()
print("  Step 3: 运行检索")
print("    python passage_retrieval.py \\")
print("      --model_name_or_path facebook/contriever-msmarco \\")
print("      --passages psgs_w100.tsv \\")
print('      --passages_embeddings "wikipedia_embeddings/*" \\')
print(f"      --data ../../{OUTPUT_FILE} \\")
print("      --output_dir ../../data/eval/popqa_retrieved/ --n_docs 20")
print()
print("  或者：使用已有的 passage_retrieval 脚本（如果环境中已有嵌入）")
sys.exit(1)
PYEOF

RESULT=$?
if [ $RESULT -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "  ✅ 下载完成！验证数据..."
    echo "============================================================"

    python -c "
import json
count = 0
total_ctxs = 0
with open('${OUTPUT_FILE}', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        count += 1
        total_ctxs += len(item.get('ctxs', []))

print(f'  总行数: {count}')
print(f'  平均每条 ctxs 数: {total_ctxs / count:.1f}')
with open('${OUTPUT_FILE}', encoding='utf-8') as f:
    first = json.loads(f.readline())
print(f'  ctxs[0] keys: {list(first[\"ctxs\"][0].keys())}')
print(f'  ctxs[0] title: {first[\"ctxs\"][0].get(\"title\", \"N/A\")[:60]}')
print(f'  ctxs[0] text[:100]: {first[\"ctxs\"][0].get(\"text\", \"N/A\")[:100]}')
"
fi
