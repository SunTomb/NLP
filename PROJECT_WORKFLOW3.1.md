# Self-RAG 课程大作业 — 项目工作流 v3.1

> **更新日期**：2026 年 4 月 25 日 20:00  
> **基于**：PROJECT_WORKFLOW3.0.md  
> **本版重点**：Generator 全量训练已完成，进入评测阶段

---

## 〇、总体进度

```
Phase 1: 基建        [███████████████████████████] 100%  ✅ 已完成
Phase 2: 复现实验    [█████████████████░░░░░░░░░░]  65%  ◀ 当前阶段
Phase 3: 改进实验    [░░░░░░░░░░░░░░░░░░░░░░░░░░░]   0%
Phase 4: 报告撰写    [░░░░░░░░░░░░░░░░░░░░░░░░░░░]   0%
```

---

## 一、已完成里程碑

| # | 里程碑 | 完成时间 | 关键结果 |
|---|--------|---------|---------|
| M0 | 环境部署 | 4/21 01:35 | conda --prefix 隔离环境，PyTorch 2.4 + vLLM 0.5.5 |
| M1 | 推理验证 | 4/21 02:30 | 7 组测试全部通过，反思 Token 行为符合论文 |
| M2 | 数据就绪 | 4/21 ~04:00 | Critic 48.5K + Generator 145K + 4 个评测集 |
| M3 | 冒烟测试 | 4/21 14:50 | Loss 8.43→5.16，单卡 A40 ~22GB |
| M4 | Critic 训练 | 4/22 12:40 | 3 epoch, Loss 13.47→0.22, 21.5h 单卡 A40 |
| M5 | Generator 冒烟 | 4/22 14:00 | 100 条冒烟通过，Adafactor + GradCkpt 确认可行 |
| **M6** | **Generator 训练** | **4/25 19:45** | **3 epoch, Loss 3.99→0.21, 66.5h 单卡 A40** |

---

### M6 Generator 训练详细报告

| 指标 | 值 |
|------|-----|
| 数据量 | 145,619 条 |
| 总步数 | 27,306 步 |
| 训练时间 | **66 小时 33 分**（约 2.8 天） |
| 平均速度 | 8.78 s/step |
| 最终 Loss | **0.21** |
| 配置 | 单卡 A40, bf16, Adafactor, GradCkpt, bs=1, accum=16, seq=2048 |
| 日志文件 | `logs/gen_train_20260423_010758.log` |
| 模型输出 | `outputs/generator_llama2_7b/` (3 shard, ~12.8GB) |
| Checkpoints | step_5000, step_10000, step_15000, step_20000, step_25000 |

**Loss 曲线摘要**：

```
Step     50   → Loss 3.99   (初始，新 special token 未训练)
Step    100   → Loss 3.17   (急剧下降 -21%)
Step    500   → Loss 1.07   (快速收敛阶段)
Step  1,000   → Loss 1.01   (Epoch 1 前期)
Step  2,000   → Loss 0.98   (进入精细学习)
Step  5,000   → Loss 0.88   (Epoch 1 中期)
--- Epoch 1 结束 (~step 9,101) ---
Step  9,100   → Loss 0.82   (Epoch 1 末尾)
Step 10,000   → Loss 0.43   (Epoch 2 开始，loss 跳降)
Step 15,000   → Loss 0.42   (Epoch 2 中期，稳定)
Step 18,000   → Loss 0.40   (Epoch 2 末尾)
--- Epoch 2 结束 (~step 18,202) ---
Step 18,200   → Loss 0.41   (Epoch 2/3 边界)
Step 20,000   → Loss 0.20   (Epoch 3 开始，再次跳降)
Step 25,000   → Loss 0.20   (Epoch 3 中期，收敛)
Step 27,000   → Loss 0.21   (Epoch 3 末尾)
Step 27,300   → Loss 0.21   (最终，总下降 94.7%)
```

**收敛分析**：
- 每个 epoch 边界有明显的 loss 阶梯式下降（0.82→0.43→0.20），符合 SFT 训练的典型模式
- 最终 loss 0.21 与 Critic 的 0.22 高度一致，说明模型已充分拟合训练数据
- 从 step 20000 到 27306，loss 在 0.19-0.22 之间波动，无过拟合迹象

**Step 9 后验证**：
```
Generator 模型加载成功 ✅
  参数量: 6.74B
  词表: 32016
  Special tokens: ['[No Retrieval]', '[Retrieval]', '[Continue to Use Evidence]', '[Irrelevant]', '[Relevant]']
```

### 遇到的工程问题汇总（详见 problem.md P13-P15）

| 编号 | 问题 | 影响 | 解决 |
|:----:|------|------|------|
| P13 | NCCL 超时（tokenize 阶段） | 4 卡 FSDP 崩溃 | `InitProcessGroupKwargs(timeout=1800s)` |
| P14 | AdamW OOM（FSDP 下） | 优化器状态 OOM | 替换为 Adafactor |
| **P15** | **FSDP 慢于单卡** | **40s/step vs 9s/step** | **改用单卡 Adafactor** |

---

## 二、下一步行动 — 复现评测（Step 10）

> [!IMPORTANT]
> Generator 模型已就绪。现在需要评测 3 个模型的性能对比：
> 1. **Our Generator**（我们复现训练的）
> 2. **Official Self-RAG**（官方预训练模型）
> 3. **Vanilla Llama 2**（基线）

### Step 10: 全面评测

#### 10.1 评测任务概览

| 任务 | 类型 | 数据文件 | 指标 |
|------|------|---------|------|
| PopQA | Short-form QA | `data/eval/popqa_longtail.jsonl` | match (EM) |
| ARC-Challenge | Multiple Choice | `data/eval/arc_challenge.jsonl` | match |
| TriviaQA | Short-form QA | `data/eval/triviaqa.jsonl` | match |
| PubHealth | Fact Verification | `data/eval/pubhealth.jsonl` | match |
| ASQA | Long-form QA | `data/eval/asqa.jsonl` | str-em, rouge-L |
| FactScore | Biography | `data/eval/factscore.jsonl` | factscore |

#### 10.2 评测命令（no_retrieval 模式）

> [!TIP]
> 先用 `no_retrieval` 模式评测（不需要检索器），比较模型自身的生成能力。后续可以加入检索器做 `retrieval` 模式对比。

```bash
source /NAS/yesh/NLP/activate.sh
cd /NAS/yesh/NLP

# === 模型 1: 我们复现的 Generator ===

# PopQA
CUDA_VISIBLE_DEVICES=5 python self-rag/retrieval_lm/run_short_form.py \
    --model_name outputs/generator_llama2_7b \
    --input_file data/eval/popqa_longtail.jsonl \
    --max_new_tokens 100 --threshold 0.2 \
    --output_file results/popqa_our.json \
    --metric match --ndocs 0 --no_retrieval

# ARC-Challenge
CUDA_VISIBLE_DEVICES=5 python self-rag/retrieval_lm/run_short_form.py \
    --model_name outputs/generator_llama2_7b \
    --input_file data/eval/arc_challenge.jsonl \
    --max_new_tokens 50 --threshold 0.2 \
    --output_file results/arc_our.json \
    --metric match --ndocs 0 --no_retrieval

# TriviaQA
CUDA_VISIBLE_DEVICES=5 python self-rag/retrieval_lm/run_short_form.py \
    --model_name outputs/generator_llama2_7b \
    --input_file data/eval/triviaqa.jsonl \
    --max_new_tokens 100 --threshold 0.2 \
    --output_file results/triviaqa_our.json \
    --metric match --ndocs 0 --no_retrieval

# PubHealth
CUDA_VISIBLE_DEVICES=5 python self-rag/retrieval_lm/run_short_form.py \
    --model_name outputs/generator_llama2_7b \
    --input_file data/eval/pubhealth.jsonl \
    --max_new_tokens 50 --threshold 0.2 \
    --output_file results/pubhealth_our.json \
    --metric match --ndocs 0 --no_retrieval


# === 模型 2: 官方 Self-RAG（基线对比） ===
# 替换 --model_name 为 models/selfrag_llama2_7b，output_file 改为 *_official.json

# === 模型 3: Vanilla Llama 2（基线） ===
# 替换 --model_name 为 models/Llama-2-7b-hf，output_file 改为 *_llama2.json
```

#### 10.3 预计结果

基于论文 Table 2（no retrieval 模式）：

| 任务 | Llama2-7B | Self-RAG (论文) | 我们的目标 |
|------|:---------:|:---------------:|:---------:|
| PopQA | ~21% | ~54% | >45% |
| ARC-C | ~45% | ~67% | >55% |
| TriviaQA | ~55% | ~69% | >60% |
| PubHealth | ~45% | ~69% | >55% |

> [!WARNING]
> 我们的训练数据是从 HuggingFace 下载的（与论文原始数据可能有细微差异），且使用了 Adafactor 而非 AdamW。结果可能与论文数值有 5-10% 偏差，属于正常范围。

---

## 三、后续步骤概览

### Step 11-12: 改进实验

| 实验 | 基座 | 数据 | 预计时间 |
|------|------|------|:--------:|
| I1: Qwen 基座 | Qwen2.5-7B | 同 Generator 数据 | 2-3 天 |
| I2: 中文数据 | Qwen2.5-7B | 中文 QA + 反思 Token | 1-2 天 |
| I3: 消融实验 | — | — | 0.5 天 |

### Step 13: 报告撰写

---

## 四、修订后时间线

```
Week:  W1(4/21)──W2(4/28)──W3(5/5)──W4(5/12)──W5(5/19)──W6(5/26)──W7(6/2)──W8(6/9)──→提交
       ├── P1: 基建 ─┤
       │ ✅ 全部完成 │
                     ├── P2: 复现训练 ───────────────────┤
                     │ ✅ Critic (21.5h)                 │
                     │ ✅ Generator (66.5h)              │
                     │ ◀ 全面评测 (1-2d)                 │
                     │   基线对比 (半天)                  │
                                                        ├── P3: 改进实验 ──────┤
                                                        │  Qwen 基座 (I1)     │
                                                        │  中文适配 (I2)       │
                                                        │  消融实验             │
                                                        │  Demo 系统            │
                                                                              ├── P4: 报告 ─┤
                                                                              │  撰写+图表   │
                                                                              │  审校+提交   │
                                                                                             └→ 提交
```

---

## 五、里程碑检查清单

| 里程碑 | 目标时间 | 判定标准 | 状态 |
|--------|---------|---------|:----:|
| M0: 环境部署 | 4/21 | conda + 依赖 + GPU | ✅ |
| M1: 推理验证 | 4/21 | 反思 Token 正确 | ✅ |
| M2: 数据就绪 | 4/21 | 全部数据到位 | ✅ |
| M3: 冒烟通过 | 4/21 | Loss 正常下降 | ✅ |
| M4: Critic 训练 | 4/22 | 3 epoch, loss=0.22 | ✅ |
| M5: Generator 冒烟 | 4/22 | 100 条无 crash | ✅ |
| **M6: Generator 训练** | **4/25** | **3 epoch, loss=0.21** | **✅** |
| **M7: 复现评测** | **W2 4/28** | **4 任务 + 3 模型对比** | **◀ 下一步** |
| M8: 改进实验 | W4-5 5/12 | Qwen / 中文 / 消融 | 🔲 |
| M9: 报告初稿 | W6 5/26 | ≥ 5000 字完整报告 | 🔲 |
| M10: 最终提交 | W8 6/15 | 报告 + 代码 + Demo | 🔲 |

---

## 六、常用命令速查

```bash
# 环境
source /NAS/yesh/NLP/activate.sh

# GPU 状态（简洁版）
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits

# tmux
tmux new -s eval          # 创建
tmux attach -t eval       # 附加
# Ctrl+B D                # 分离

# 监控评测
tail -f results/*.json
```

---

## 七、文件清单

```
/NAS/yesh/NLP/
├── self-rag/
│   ├── data_creation/
│   │   └── train_special_tokens.py   # Critic 训练 ✅
│   └── retrieval_lm/
│       ├── finetune.py               # Generator 训练 ✅
│       ├── run_short_form.py         # Short-form 评测 ← 下一步使用
│       └── run_long_form_static.py   # Long-form 评测
├── data/
│   ├── critic/
│   │   ├── critic_train_data_train.json  # 48.5K 条   ✅
│   │   └── critic_smoke_test.json        # 100 条
│   ├── generator/                        # 145K 条     ✅
│   │   └── train.jsonl
│   └── eval/                             # 评测数据集  ← 下一步使用
│       ├── popqa_longtail.jsonl
│       ├── arc_challenge.jsonl
│       ├── triviaqa.jsonl
│       └── pubhealth.jsonl
├── outputs/
│   ├── critic_smoke/              # 冒烟产出         ✅
│   ├── critic_llama2_7b/          # 正式 Critic      ✅
│   ├── gen_smoke/                 # Generator 冒烟   ✅
│   └── generator_llama2_7b/       # 正式 Generator   ✅ (6.74B, 32016 vocab)
│       ├── model-00001-of-00003.safetensors  (4.6GB)
│       ├── model-00002-of-00003.safetensors  (4.6GB)
│       ├── model-00003-of-00003.safetensors  (3.3GB)
│       ├── tokenizer.json + tokenizer.model
│       ├── config.json + generation_config.json
│       └── step_{5000,10000,15000,20000,25000}/  # checkpoints
├── results/                       # 评测结果  ← 下一步生成
├── models/                        # 符号链接
│   ├── selfrag_llama2_7b → HF Cache
│   ├── Llama-2-7b-hf → HF Cache
│   └── Qwen2.5-7B → HF Cache            # 改进实验（待下载）
└── logs/
    ├── critic_smoke_*.log          # 冒烟日志
    ├── critic_train_*.log          # Critic 训练日志   ✅
    ├── gen_smoke_*.log             # Generator 冒烟    ✅
    └── gen_train_20260423_010758.log  # Generator 训练  ✅ (66.5h)
```

---

> **立即行动**：
> 1. 确认评测数据集存在且格式正确
> 2. 运行 PopQA no_retrieval 评测（Step 10.2）
> 3. 逐个评测 4 个任务 × 3 个模型
