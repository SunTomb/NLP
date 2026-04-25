# PROJECT_WORKFLOW3.2 — PopQA always_retrieve 评测

> **更新时间**：2026-04-26 05:07  
> **依赖版本**：`PROJECT_WORKFLOW3.1.md`（Step 10 全面评测已完成）

---

## 背景回顾

### Step 10.3 评测结果（no_retrieval, 14267 条）

| 任务 | 指标 | Our (微调) | Official | Llama2 |
|------|------|:---:|:---:|:---:|
| PopQA | match | 100.00% ⚠️ | 99.99% ⚠️ | 100.00% ⚠️ |
| ARC-C | match | 57.25% ✅ | 62.29% ✅ | 43.34% ✅ |
| TriviaQA | match | 31.50% ✅ | 29.75% ✅ | 17.05% ✅ |

### 问题链

- **P19**: `match()` 子串匹配在长文本上虚假 100%
- **P20**: 原始数据 `ctxs` 字段为空列表，导致 always_retrieve 退化为 no_retrieval

---

## Step 10.5: 数据获取 ✅

### 最终数据来源

| 来源 | `selfrag/selfrag_train_data` | `awinml/popqa_longtail_w_gs` |
|------|:---:|:---:|
| 状态 | ❌ 无 PopQA 评测文件 | ✅ 下载成功 |
| 样本数 | — | **1,399** |
| ctxs/条 | — | **25** |
| ctxs 字段 | — | `id, title, text, score` |

> **注意**：下载的数据 (1,399 条) 是原始 14,267 条的子集。为确保公平对比，**两种模式 (no_retrieval + always_retrieve) 均在同一 1,399 条数据上重新评测**。

### 数据文件

```
data/eval/popqa_longtail_w_gs.jsonl         # 1,399 条 (含 25 ctxs/条) ← 当前
data/eval/popqa_longtail_w_gs.jsonl.bak_no_ctxs  # 14,267 条 (ctxs 为空) 备份
```

---

## Step 10.6: 完整对比评测

### 评测方案

**同一数据集 (1,399 条) 上的 4 组实验**：

| 实验 | 模型 | 模式 | 输出文件 |
|:---:|------|------|------|
| 1 | Our (微调) | no_retrieval | `popqa_nr_our.json` |
| 2 | Official | no_retrieval | `popqa_nr_official.json` |
| 3 | Our (微调) | always_retrieve | `popqa_ar_our.json` |
| 4 | Official | always_retrieve | `popqa_ar_official.json` |

### 预计资源

| 项目 | 估算 |
|------|:---:|
| no_retrieval prompts | 1,399 × 2 模型 |
| always_retrieve prompts | 1,399 × 5 docs × 2 模型 = **13,990** |
| 总 prompts | ~16,788 |
| 显存占用 | ~35-40 GB |
| 总耗时 | **~15-20 分钟** |

### 执行命令

```bash
source /NAS/yesh/NLP/activate.sh
cd /NAS/yesh/NLP

# 确认 GPU 空闲
nvidia-smi

# 运行完整对比评测 (no_retrieval + always_retrieve)
tmux new -s eval_popqa
CUDA_VISIBLE_DEVICES=7 bash scripts/run_eval_popqa_full.sh
```

### 预期结果

参考论文 Table 2：

| 模型 | no_retrieval | always_retrieve |
|------|:---:|:---:|
| Self-RAG (论文) | 32.7% | **54.9%** |
| Llama2 (论文) | 21.7% | N/A |
| Our (微调) | 待评测 | 待评测 |
| Official | 待评测 | 待评测 |

> 预期：always_retrieve >> no_retrieval（检索段落显著提升问答准确率）

---

## 结果整合（待填写）

评测完成后更新最终对比表格：

| 任务 | 模式 | 数据量 | Our (微调) | Official | 论文值 |
|------|------|:---:|:---:|:---:|:---:|
| PopQA | no_retrieval | 1,399 | — | — | 32.7% |
| PopQA | always_retrieve | 1,399 | — | — | 54.9% |
| ARC-C | no_retrieval | 1,172 | 57.25% | 62.29% | — |
| TriviaQA | no_retrieval | 2,000 | 31.50% | 29.75% | — |

---

## 文件结构

```
NLP/
├── self-rag/retrieval_lm/
│   ├── run_eval_batch.py               # no_retrieval 批量推理 ✅
│   └── run_eval_batch_retrieve.py      # always_retrieve 批量推理 ✅ (含 WARNING)
├── scripts/
│   ├── run_eval_all.sh                 # 全面评测 (no_retrieval) ✅
│   ├── run_eval_popqa_full.sh          # PopQA 完整对比评测 ← 新增
│   └── download_popqa_with_ctxs.sh     # 数据下载脚本 ✅
├── data/eval/
│   ├── popqa_longtail_w_gs.jsonl       # 1,399 条 (含 25 ctxs) ✅
│   └── popqa_longtail_w_gs.jsonl.bak_no_ctxs  # 14,267 条 (备份)
└── results/
    ├── popqa_nr_{our,official}.json     # no_retrieval (1,399 条) ← 待生成
    └── popqa_ar_{our,official}.json     # always_retrieve (1,399 条) ← 待生成
```

---

> **立即行动**：
>
> ```bash
> CUDA_VISIBLE_DEVICES=<空闲GPU> bash scripts/run_eval_popqa_full.sh
> ```
>
> 等待 ~15-20 分钟，查看日志中的最终对比表格。
