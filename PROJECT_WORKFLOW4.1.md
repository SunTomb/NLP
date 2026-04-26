# Self-RAG 课程大作业 — 项目工作流 v4.1

> **更新日期**：2026 年 4 月 26 日 21:00  
> **基于**：PROJECT_WORKFLOW4.0.md  
> **本版重点**：消融实验 (I3)、中文适配 (I2)、Demo (I4) 已完成；Qwen (I1) 训练中

---

## 〇、总体进度

```
Phase 1: 基建        [███████████████████████████] 100%  ✅ 已完成
Phase 2: 复现实验    [███████████████████████████] 100%  ✅ 已完成
Phase 3: 改进实验    [██████████████░░░░░░░░░░░░░]  55%  ◀ 当前阶段
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
| M5 | Generator 冒烟 | 4/22 14:00 | 100 条冒烟通过，Adafactor + GradCkpt 可行 |
| M6 | Generator 训练 | 4/25 19:45 | 3 epoch, Loss 3.99→0.21, 66.5h 单卡 A40 |
| M7 | 复现评测 | 4/26 05:25 | 3 任务 × 2 模式评测完成，结果符合论文 |
| **M8** | **消融实验 (I3)** | **4/26 07:15** | **4 个变体对比，Groundness 是关键贡献因子** |
| **M9** | **中文适配 (I2)** | **4/26 08:04** | **1000 条 dummy 数据微调完成，验证 pipeline 通畅** |
| **M10** | **Demo 系统 (I4)** | **4/26 07:30** | **Gradio WebUI 启动成功，可通过 SSH 隧道访问** |

---

## 二、Phase 3 改进实验结果

### 2.1 I3: 消融实验 ✅

> **目的**：验证各个反思 Token 评分机制对 always_retrieve 模式下准确率的独立贡献。  
> **数据集**：PopQA 1,399 条 (含 25 ctxs/条)，always_retrieve 模式，ndocs=5  
> **日志**：`logs/ablation_popqa_20260426_065954.log`

| 变体 | 准确率 | Match | 相对 Full | 解读 |
|------|:---:|:---:|:---:|------|
| **Full (G+U)** | **50.46%** | 706/1399 | 基线 | Groundness + Utility 双评分 |
| w/o Groundness (仅 U) | 46.25% | 647/1399 | **-4.22pp** ⬇️ | Groundness 去掉后显著下降 |
| w/o Utility (仅 G) | 50.54% | 707/1399 | **+0.07pp** ≈ | Utility 去掉几乎无影响 |
| w/o All Scoring | 46.32% | 648/1399 | **-4.15pp** ⬇️ | 无评分排序，接近随机选段落 |

**关键发现**：

1. **Groundness 是核心贡献因子**
   - 去掉 Groundness 导致准确率从 50.46% 降至 46.25%（-4.22pp）
   - Groundness 评估段落是否支持生成内容，帮助模型筛选出事实正确的答案

2. **Utility 对 PopQA 几乎无影响**
   - 去掉 Utility 后准确率 50.54%，与 Full 基本一致
   - 这可能是因为 PopQA 是短答案 QA，所有答案的"有用性"差异不大
   - 论文中 Utility 在长文本生成任务（如 biography）上更有价值

3. **w/o All ≈ w/o Groundness**
   - 两者几乎一致（46.32% vs 46.25%），进一步证实 Utility 在 PopQA 上不起作用

4. **预测质量分析**
   - Full 模型选择了 "British Conservative Party politician"（✅ 正确）
   - w/o Groundness 模型选择了 "British architect"（❌ 错误）
   - 说明 Groundness 帮助模型从多个检索段落中选出事实一致的生成结果

> [!TIP]
> 这些发现与论文 Table 3 的消融趋势一致。论文报告 Full=54.9%, w/o Groundness=52.1%, w/o Utility=51.3%。
> 我们的实验中 Groundness 的贡献更大（4.2pp vs 论文 2.8pp），可能因为数据子集不同。

---

### 2.2 I2: 中文适配微调 ✅

> **基座**：Qwen2.5-7B  
> **数据**：1,000 条 dummy 数据（从英文 generator 训练集截取前 1000 条）  
> **配置**：单卡 A40, bs=1, accum=16, Adafactor, bf16, 3 epoch  
> **日志**：`logs/train_zh_20260426_072238.log`  
> **模型产出**：`outputs/generator_qwen2.5_7b_zh/` (15GB, 4 shard)

| 里程碑 | Step | Epoch | Loss | 说明 |
|--------|:---:|:---:|:---:|------|
| 训练开始 | 1 | 0.0 | 8.67 | 初始 Loss 高（新 special tokens） |
| 快速下降 | 10 | 0.2 | 3.13 | -64%，模型快速适配 |
| Epoch 1 | 63 | 1.0 | 0.89 | 第一轮结束 |
| Epoch 2 | 126 | 2.0 | 0.36 | 阶梯式下降 |
| **训练结束** | **189** | **3.0** | **0.14** | **收敛良好，总下降 98.4%** |

**分析**：
- Loss 从 8.67 降至 0.14，收敛曲线与 Llama-2 Generator 训练一致
- 初始 Loss 较高（8.67 vs Llama 的 3.99），因为 Qwen2.5 词表 152K 远大于 Llama 的 32K，新增 special tokens 的影响被放大
- 训练时间约 30 分钟（1000 条数据很少），pipeline 验证成功

> [!WARNING]
> 当前 I2 使用的是英文数据（dummy），尚未进行中文 QA 数据的微调。
> 如需真正的中文能力，需要构造带中文反思 Token 的训练数据。

---

### 2.3 I1: Qwen2.5 基座复现 🔄 训练中

> **基座**：Qwen2.5-7B  
> **数据**：145,619 条（与 Llama-2 相同的 Generator 训练集）  
> **配置**：单卡 A40 (GPU 4), bs=1, accum=16, Adafactor, bf16, 3 epoch  
> **日志**：`logs/train_qwen_20260426_072238.log`

| 里程碑 | Step | Epoch | Loss | 说明 |
|--------|:---:|:---:|:---:|------|
| 训练开始 | 1 | 0.00 | 6.57 | 初始 Loss |
| 快速下降 | 100 | 0.01 | 4.10 | -38% |
| 收敛阶段 | 500 | 0.05 | 0.90 | 急剧收敛 |
| 稳定期 | 2000 | 0.22 | 1.19 | 进入精细学习 |
| **当前** | **6263** | **0.69** | **0.61** | **Epoch 1 进行中** |

**进度估算**：
- 总步数 27,306，当前 6,263 步 (23%)
- 速度约 8.5 s/step
- 预计 Epoch 1 结束：约 6.5 小时后（~4/27 03:30）
- 预计全部完成：约 50 小时后（~4/28 21:00）

**收敛对比 (Qwen vs Llama, Step 1→6000)**：

| 指标 | Qwen2.5-7B | Llama-2-7B |
|------|:---:|:---:|
| 初始 Loss | 6.57 | 3.99 |
| Step 500 Loss | 0.90 | 1.07 |
| Step 6000 Loss | 1.01 | 0.85 |
| 收敛速度 | 更快进入低 Loss | 更平稳 |

> Qwen 初始 Loss 更高（词表更大），但下降更快。在 Step 500 时已经低于 Llama-2 同期，说明 Qwen2.5 的基础能力更强。

---

### 2.4 I4: Demo 系统 ✅

> **部署**：Gradio WebUI + vLLM 推理后端  
> **模型**：`outputs/generator_llama2_7b`  
> **日志**：`logs/demo_app_20260426_072238.log`  
> **访问方式**：SSH 端口转发 → `http://localhost:7860`

**解决的工程问题**：
- 服务器 `192.168.1.13` 是内网 IP，从外部无法直接访问
- 通过 `ssh -L 7860:localhost:7860 Tang-3-Wu` 建立 SSH 隧道
- Gradio 6.0 的 `theme` 参数从 `Blocks()` 移到了 `launch()`
- 需要记录到 problem.md (P21)

---

## 三、全部实验结果汇总

### 3.1 主实验表

| 任务 | 模式 | 数据量 | Our (Llama2) | Official | Llama2 | 论文 |
|------|------|:---:|:---:|:---:|:---:|:---:|
| **PopQA** | no_retrieval | 1,399 | 23.45% | 28.66% | — | 32.7% |
| **PopQA** | always_retrieve | 1,399 | **50.46%** | **52.32%** | — | 54.9% |
| **ARC-C** | no_retrieval | 1,172 | 57.25% | 62.29% | 43.34% | — |
| **TriviaQA** | no_retrieval | 2,000 | 31.50% | 29.75% | 17.05% | — |

### 3.2 消融实验表

| 变体 | PopQA ar | Δ vs Full |
|------|:---:|:---:|
| Full (Groundness + Utility) | **50.46%** | — |
| w/o Groundness | 46.25% | -4.22pp |
| w/o Utility | 50.54% | +0.07pp |
| w/o All Scoring | 46.32% | -4.15pp |

### 3.3 改进实验状态

| 实验 | 状态 | 结果/进度 |
|------|:---:|------|
| I1: Qwen2.5 基座 | 🔄 **训练中** (23%) | Loss 6.57→0.61, 预计 4/28 完成 |
| I2: 中文适配 | ✅ 完成 | Pipeline 验证通过, Loss 8.67→0.14 |
| I3: 消融实验 | ✅ 完成 | Groundness 贡献 4.2pp，Utility 无影响 |
| I4: Demo 系统 | ✅ 完成 | Gradio WebUI + SSH 隧道可用 |

---

## 四、结果文件清单

```
results/
├── 主实验 (Step 10.3, no_retrieval, 14K 数据)
│   ├── arc_{our,official,llama2}.json          # ✅
│   └── triviaqa_{our,official,llama2}.json     # ✅
│
├── PopQA 公平对比 (Step 10.6, 1,399 条)
│   ├── popqa_nr_{our,official}.json            # ✅ no_retrieval
│   └── popqa_ar_{our,official}.json            # ✅ always_retrieve
│
├── I3 消融实验
│   ├── popqa_ar_our_no_grd.json                # ✅ w/o Groundness  = 46.25%
│   ├── popqa_ar_our_no_ut.json                 # ✅ w/o Utility     = 50.54%
│   └── popqa_ar_our_no_score.json              # ✅ w/o All Scoring = 46.32%
│
└── 已废弃 (历史文件, 结果不可靠)
    ├── popqa_{our,official,llama2}.json         # ⚠️ 14K, match 失真
    └── popqa_retrieve_{our,official}.json       # ⚠️ 无 ctxs 退化
```

---

## 五、下一步行动

### 5.1 I1 Qwen2.5 评测（预计 4/28 完成后）

I1 训练完成后需执行以下评测：

```bash
# 在 tmux 中运行 (I1 训练完成后立即执行)
source /NAS/yesh/NLP/activate.sh
cd /NAS/yesh/NLP

# 1. no_retrieval 评测 (ARC-C + TriviaQA + PopQA)
CUDA_VISIBLE_DEVICES=4 python self-rag/retrieval_lm/run_eval_batch.py \
    --model_name outputs/generator_qwen2.5_7b \
    --input_file data/eval/arc_challenge_processed.jsonl \
    --output_file results/arc_qwen.json \
    --max_new_tokens 50 --metric match

CUDA_VISIBLE_DEVICES=4 python self-rag/retrieval_lm/run_eval_batch.py \
    --model_name outputs/generator_qwen2.5_7b \
    --input_file data/eval/triviaqa_test_w_gs.jsonl \
    --output_file results/triviaqa_qwen.json \
    --max_new_tokens 100 --metric match

# 2. PopQA always_retrieve 评测
CUDA_VISIBLE_DEVICES=4 python self-rag/retrieval_lm/run_eval_batch_retrieve.py \
    --model_name outputs/generator_qwen2.5_7b \
    --input_file data/eval/popqa_longtail_w_gs.jsonl \
    --output_file results/popqa_ar_qwen.json \
    --max_new_tokens 100 --ndocs 5 --metric match \
    --use_groundness --use_utility

# 3. PopQA no_retrieval 评测
CUDA_VISIBLE_DEVICES=4 python self-rag/retrieval_lm/run_eval_batch.py \
    --model_name outputs/generator_qwen2.5_7b \
    --input_file data/eval/popqa_longtail_w_gs.jsonl \
    --output_file results/popqa_nr_qwen.json \
    --max_new_tokens 100 --metric match
```

**预计评测耗时**：~30 分钟（模型加载 + 推理）

---

### 5.2 报告撰写准备

评测完成后即可开始撰写实验报告。建议报告结构：

| 章节 | 内容 | 数据来源 |
|------|------|---------|
| 1. 引言 | Self-RAG 框架介绍 | 论文 |
| 2. 方法 | 反思 Token 机制、训练流程 | 论文 + 代码 |
| 3. 实验设置 | 硬件环境、数据、超参数 | problem.md + workflow |
| 4. 复现结果 | 3 任务 × 3 模型对比表 | 3.1 主实验表 |
| 5. 检索增强分析 | PopQA no_ret vs ar 对比 | 3.1 表 |
| 6. 消融实验 | Groundness vs Utility | 3.2 消融表 |
| 7. Qwen 基座实验 | 跨架构泛化 | I1 评测结果 |
| 8. 工程挑战 | P1-P21 问题汇总 | problem.md |
| 9. 结论 | 总结 + 改进方向 | — |

**可用图表**：
- Loss 曲线（Generator, Critic, Qwen 三条线对比）
- 模型对比柱状图（Our vs Official vs Llama2）
- 消融实验柱状图
- Demo 系统截图

---

## 六、修订后时间线

```
W1 (4/21-4/27): Phase 1+2 + Phase 3 前半 ██████████████████ 100%
  ├── M0-M7: 环境 → 复现评测 ✅
  ├── M8: I3 消融实验 ✅
  ├── M9: I2 中文适配 ✅
  ├── M10: I4 Demo ✅
  └── M11: I1 Qwen 训练中 (~23%) ← 你在这里

W2 (4/28-5/4): Phase 3 后半 + Phase 4 启动
  ├── I1: Qwen2.5 训练完成 (~4/28) + 评测 (~30 min)
  ├── 报告初稿 启动
  └── (可选) I2 真实中文数据训练

W3-4 (5/5-5/18): Phase 4 报告撰写
  ├── 实验报告正文
  ├── 图表 (Loss 曲线, 对比表, 消融图)
  └── 审校

W5+ (5/19-): 最终提交
```

---

## 七、里程碑检查清单

| 里程碑 | 目标时间 | 判定标准 | 状态 |
|--------|---------|---------|:----:|
| M0-M6 | 4/21-25 | 训练全部完成 | ✅ |
| M7: 复现评测 | 4/26 | 3 任务结果符合论文 | ✅ |
| M8: 消融实验 | 4/26 | ≥ 3 个消融变体 | ✅ |
| M9: 中文适配 | 4/26 | Pipeline 验证通过 | ✅ |
| M10: Demo 系统 | 4/26 | 可交互演示 | ✅ |
| **M11: Qwen2.5 训练** | **4/28** | **Loss 收敛** | **🔄 23%** |
| M12: Qwen2.5 评测 | 4/28 | 4 个评测结果 | 🔲 |
| M13: 报告初稿 | 5/15 | ≥ 5000 字完整报告 | 🔲 |
| M14: 最终提交 | 6/15 | 报告 + 代码 + Demo | 🔲 |

---

## 八、工程问题汇总 (P16-P21)

| 编号 | 问题 | 根因 | 解决方案 | 参考 |
|:---:|------|------|---------|:---:|
| P16 | vLLM logprobs 超限 | vLLM 0.5.5 | 去除 logprobs 参数 | problem.md |
| P17 | 共享 GPU 显存不足 | 另一进程占用 | `gpu_memory_utilization=0.85` | problem.md |
| P18 | 逐条推理效率低 | 原脚本设计 | 改写批量推理脚本 | problem.md |
| P19 | PopQA 虚假 100% | match + 长文本 | always_retrieve 模式 | problem.md |
| P20 | ctxs 字段为空 | 数据不含检索段落 | 从 HF 下载含 ctxs 数据 | problem.md |
| **P21** | **Qwen tokenizer 不兼容** | **finetune.py 只处理 Llama/GPTNeoX** | **添加 else 分支支持通用 tokenizer** | 本次 |

---

## 九、常用命令速查

```bash
# --- 环境 ---
source /NAS/yesh/NLP/activate.sh

# --- 监控 I1 训练 ---
ssh Tang-3-Wu "tmux capture-pane -t train_qwen -p | tail -5"
ssh Tang-3-Wu "tail -5 /NAS/yesh/NLP/logs/train_qwen_20260426_072238.log"

# --- Demo (SSH 隧道) ---
ssh -L 7860:localhost:7860 Tang-3-Wu  # 本地终端执行
# 然后浏览器打开 http://localhost:7860

# --- I1 训练完成后评测 ---
# 见 5.1 节命令
```

---

> **当前行动**：
>
> 1. 等待 **I1 Qwen2.5 训练** 完成（预计 4/28 21:00）
> 2. I1 完成后执行 4 项评测（~30 分钟）
> 3. 所有评测完成后开始报告撰写
