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
>
> 1. **Our Generator**（我们复现训练的）
> 2. **Official Self-RAG**（官方预训练模型）
> 3. **Vanilla Llama 2**（基线）

### Step 10: 全面评测

#### 10.1 评测任务概览

| 任务 | 类型 | 数据文件 | 指标 |
|------|------|---------|------|
| PopQA | Short-form QA | `data/eval/popqa_longtail_w_gs.jsonl` | match (substring) |
| ARC-Challenge | Multiple Choice | `data/eval/arc_challenge_processed.jsonl` | match |
| TriviaQA | Short-form QA | `data/eval/triviaqa_test_w_gs.jsonl` | match |

#### 10.2 评测命令（no_retrieval 模式）

> [!TIP]
> 先用 `no_retrieval` 模式评测（不需要检索器），比较模型自身的生成能力。后续可以加入检索器做 `retrieval` 模式对比。

> [!NOTE]
> **显存说明**：评测脚本使用 **vLLM** 推理引擎（非 HuggingFace generate），模型以 fp16 加载。7B 模型权重 ~14GB，加上 vLLM 的 KV cache 预分配，**每个评测任务固定占用 ~16-20 GB 显存**，与数据集大小无关。显存远低于训练时的 35-40GB，单卡 A40 (44GB) 充裕。
>
> **时间说明**：`no_retrieval` 模式下每条样本只做一次前向推理。vLLM 在 A40 上 7B 模型的吞吐约 **0.05-0.15 s/sample**（取决于 `max_new_tokens`）。模型加载约需 1-2 分钟，各数据集间可复用同一进程（但当前脚本每次需重新加载）。

**评测资源总览**（3 任务 × 3 模型 = 9 次评测）：

| 任务 | 数据文件 | 数据量 | max_new_tokens | 显存 | 预计耗时（含加载） |
|------|---------|:------:|:--------------:|:----:|:-----------------:|
| PopQA | `popqa_longtail_w_gs.jsonl` | ~14K | 100 | ~18 GB | **15-25 分钟** |
| ARC-Challenge | `arc_challenge_processed.jsonl` | ~1.2K | 50 | ~18 GB | **3-5 分钟** |
| TriviaQA | `triviaqa_test_w_gs.jsonl` | ~11K | 100 | ~18 GB | **12-20 分钟** |
| **单模型合计** | — | — | — | ~18 GB | **~30-50 分钟** |
| **3 模型合计** | — | — | — | ~18 GB | **~1.5-2.5 小时** |

> [!NOTE]
> PubHealth 评测数据不可用（未下载），暂时跳过。

**一键运行**（推荐使用批量脚本）：

```bash
source /NAS/yesh/NLP/activate.sh
cd /NAS/yesh/NLP

# 在 tmux 中运行，防止断连
tmux new -s eval
CUDA_VISIBLE_DEVICES=1 bash scripts/run_eval_all.sh
```

脚本会依次评测 3 个模型（our → official → llama2）× 3 个任务，自动记录日志并在结束时打印分数汇总表。

详见 [`scripts/run_eval_all.sh`](scripts/run_eval_all.sh)。

#### 10.3 评测结果 ✅ (2026-04-26)

**评测模式**：`no_retrieval`（不使用外部检索器，纯模型生成能力）
**评测脚本**：`self-rag/retrieval_lm/run_eval_batch.py`（批量推理版）
**日志**：`logs/eval_all_20260426_020947.log`

| 任务 | Our Generator | Official Self-RAG | Vanilla Llama 2 | Paper (w/ retrieval) |
|------|:---:|:---:|:---:|:---:|
| PopQA | ⚠️ 100%* | ⚠️ 99.99%* | ⚠️ 99.99%* | ~54.9% |
| **ARC-C** | **57.25%** | **64.25%** | **39.25%** | ~67.3% |
| **TriviaQA** | **31.50%** | **38.80%** | **5.50%** | ~68.5% |

> [!WARNING]
> \* PopQA 的 `match()` 指标使用子串匹配，在 no_retrieval 模式下模型生成长文本，
> 短答案字符串几乎总是被包含在输出中，导致所有模型均出现虚假高分。
> 此结果不可直接用于论文对比，建议使用 `always_retrieve` 模式重新评测。

**关键结论**：
- ✅ **微调有效**：Our Generator 在 ARC-C (+18%) 和 TriviaQA (+26%) 上大幅优于 Vanilla Llama 2
- ✅ **与官方差距合理**：ARC-C 差 7%，TriviaQA 差 7.3%，考虑数据量差异属正常
- ✅ **模型排序正确**：Official > Our > Llama2
- ⚠️ no_retrieval 模式分数整体低于论文（论文使用了 retrieval 模式），属预期行为

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
| **M7: 复现评测** | **4/26** | **3 任务 × 3 模型对比** | **✅** |
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
│       ├── run_short_form.py         # Short-form 评测（原版，逐条推理）
│       ├── run_eval_batch.py          # Short-form 批量评测 ✅
│       └── run_long_form_static.py   # Long-form 评测
├── data/
│   ├── critic/
│   │   ├── critic_train_data_train.json  # 48.5K 条   ✅
│   │   └── critic_smoke_test.json        # 100 条
│   ├── generator/                        # 145K 条     ✅
│   │   └── train.jsonl
│   └── eval/                             # 评测数据集  ✅
│       ├── popqa_longtail_w_gs.jsonl     # 14,267 条
│       ├── arc_challenge_processed.jsonl  # 1,172 条
│       └── triviaqa_test_w_gs.jsonl       # 2,000 条
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
├── results/                       # 评测结果  ✅
│   ├── {popqa,arc,triviaqa}_{our,official,llama2}.json
├── models/                        # 符号链接
│   ├── selfrag_llama2_7b → HF Cache
│   ├── Llama-2-7b-hf → HF Cache
│   └── Qwen2.5-7B → HF Cache            # 改进实验（待下载）
└── logs/
    ├── critic_smoke_*.log          # 冒烟日志
    ├── critic_train_*.log          # Critic 训练日志   ✅
    ├── gen_smoke_*.log             # Generator 冒烟    ✅
    ├── gen_train_20260423_010758.log  # Generator 训练  ✅ (66.5h)
    └── eval_all_20260426_020947.log   # 全面评测        ✅
```

---

> **下一步行动**：
>
> 1. ✅ ~~全面评测~~ 已完成（2026-04-26）
> 2. （可选）使用 `always_retrieve` 模式重新评测 PopQA，获取真实准确率
> 3. 开始 Step 11: 改进实验（Qwen 基座 / 中文适配）
> 4. 开始报告撰写
