# PROJECT_WORKFLOW 4.2 — Qwen2.5 训练完成 & 评测进行中

> 更新时间: 2026-04-29 15:43  
> 上一版: PROJECT_WORKFLOW4.1.md

---

## 〇、总体进度

```
Phase 1: 基建        [███████████████████████████] 100%  ✅
Phase 2: 复现实验    [███████████████████████████] 100%  ✅
Phase 3: 改进实验    [████████████████████████░░░]  90%  ◀ 当前阶段
Phase 4: 报告撰写    [████████████░░░░░░░░░░░░░░░]  40%  (初稿已完成)
```

---

## 一、本次更新 (4.1 → 4.2)

### 1.1 I1: Qwen2.5-7B 训练 ✅ 完成

| 阶段 | 时间 | 详情 |
|------|------|------|
| 原始训练 | 4/26 07:22 → 4/28 09:38 | Step 1→21,608 (79%), Tang-3 GPU 4 |
| **中断** | 4/28 09:38 | NAS 磁盘满 `No space left on device` |
| 恢复训练 | 4/28 20:24 → 4/29 14:50 | Step 18,205→27,306, Tang-2 GPU 3 |
| **完成** | **4/29 14:50** | **3 epoch, 27,306 步, 总耗时 ~69h** |

**训练数据总结**：

| 指标 | Qwen2.5-7B | Llama-2-7B |
|------|:---:|:---:|
| 初始 Loss | 6.57 | 3.99 |
| Step 500 Loss | 0.90 | 0.20 |
| 最终 Loss | **0.05** | 0.21 |
| 末 500 步平均 Loss | **0.195** | 0.474 |
| 总训练时间 | ~69h | ~66.5h |

> Qwen2.5 初始 Loss 更高（词表 152K vs 32K），但最终 Loss 显著更低（0.05 vs 0.21），说明 Qwen2.5 的预训练基础更强。

**工程问题记录**：
- P22: NAS 磁盘满导致训练中断 → 使用 `--resume_from_checkpoint epoch_1` 从 checkpoint 恢复
- P23: Tang-2 CPU 被占满导致宕机 → 服务器自动恢复，训练进度未丢失

### 1.2 Qwen2.5 评测 🔄 进行中

评测已在 Tang-3 GPU 2 的 tmux `eval_qwen` 中启动，包含 4 项任务：

| # | 任务 | 模式 | 输出文件 | 预计耗时 |
|---|------|------|---------|---------|
| 1 | PopQA | no_retrieval | `popqa_nr_qwen.json` | ~3 min |
| 2 | PopQA | always_retrieve | `popqa_ar_qwen.json` | ~10 min |
| 3 | ARC-C | no_retrieval | `arc_qwen.json` | ~3 min |
| 4 | TriviaQA | no_retrieval | `triviaqa_qwen.json` | ~5 min |

**日志**: `logs/eval_qwen_*.log`

---

## 二、已完成里程碑

| # | 里程碑 | 完成时间 | 关键结果 |
|---|--------|---------|---------|
| M0 | 环境部署 | 4/21 01:35 | conda --prefix 隔离环境 |
| M1 | 推理验证 | 4/21 02:30 | 7 组测试全部通过 |
| M2 | 数据就绪 | 4/21 ~04:00 | Critic 48.5K + Generator 145K |
| M3 | 冒烟测试 | 4/21 14:50 | 单卡 A40 ~22GB |
| M4 | Critic 训练 | 4/22 12:40 | Loss 13.47→0.22, 21.5h |
| M5 | Generator 冒烟 | 4/22 14:00 | Adafactor + GradCkpt 可行 |
| M6 | Generator 训练 | 4/25 19:45 | Loss 3.99→0.21, 66.5h |
| M7 | 复现评测 | 4/26 05:25 | 3 任务 × 2 模式评测完成 |
| M8 | 消融实验 (I3) | 4/26 07:15 | Groundness 贡献 4.22pp |
| M9 | 中文适配 (I2) | 4/26 08:04 | Pipeline 验证通过 |
| M10 | Demo 系统 (I4) | 4/26 07:30 | Gradio WebUI 可用 |
| **M11** | **Qwen2.5 训练 (I1)** | **4/29 14:50** | **Loss 6.57→0.05, 69h** |
| **M12** | **报告初稿** | **4/26 23:30** | **report.tex + report.md + 5 张图** |

---

## 三、全部实验结果汇总

### 3.1 主实验表

| 任务 | 模式 | Our (Llama2) | Official | Llama2 | 论文 |
|------|------|:---:|:---:|:---:|:---:|
| **PopQA** | no_retrieval | 23.45% | 28.66% | — | 32.7% |
| **PopQA** | always_retrieve | **50.46%** | **52.32%** | — | 54.9% |
| **ARC-C** | no_retrieval | 57.25% | 62.29% | 43.34% | — |
| **TriviaQA** | no_retrieval | 31.50% | 29.75% | 17.05% | — |

### 3.2 消融实验表

| 变体 | PopQA ar | Δ vs Full |
|------|:---:|:---:|
| Full (G+U) | **50.46%** | — |
| w/o Groundness | 46.25% | −4.22pp |
| w/o Utility | 50.54% | +0.07pp |
| w/o All Scoring | 46.32% | −4.15pp |

### 3.3 改进实验状态

| 实验 | 状态 | 结果/进度 |
|------|:---:|------|
| I1: Qwen2.5 基座 | ✅ **训练完成** | Loss 6.57→0.05，🔄 评测进行中 |
| I2: 中文适配 | ✅ 完成 | Pipeline 验证通过, Loss 8.67→0.14 |
| I3: 消融实验 | ✅ 完成 | Groundness 贡献 4.22pp |
| I4: Demo 系统 | ✅ 完成 | Gradio WebUI + SSH 隧道可用 |

### 3.4 Qwen2.5 训练曲线对比

| 指标 | Qwen2.5-7B | Llama-2-7B | 分析 |
|------|:---:|:---:|------|
| Epoch 0 平均 Loss | 1.074 | — | 快速下降阶段 |
| Epoch 1 平均 Loss | 0.483 | — | 稳步收敛 |
| Epoch 2 平均 Loss | **0.200** | — | 持续优化 |
| 末 500 步平均 | **0.195** | 0.474 | Qwen 收敛更好 |

---

## 四、后续行动计划

### Step 1: 等待 Qwen2.5 评测完成 (~20min)

```bash
# 监控进度
ssh Tang-3-Wu "tmux capture-pane -t eval_qwen -p | tail -10"

# 查看日志
ssh Tang-3-Wu "ls -lt /NAS/yesh/NLP/logs/eval_qwen_*.log | head -1"
```

### Step 2: 下载评测结果到本地

```bash
scp Tang-3-Wu:/NAS/yesh/NLP/results/popqa_nr_qwen.json results/
scp Tang-3-Wu:/NAS/yesh/NLP/results/popqa_ar_qwen.json results/
scp Tang-3-Wu:/NAS/yesh/NLP/results/arc_qwen.json results/
scp Tang-3-Wu:/NAS/yesh/NLP/results/triviaqa_qwen.json results/
scp Tang-3-Wu:/NAS/yesh/NLP/logs/eval_qwen_*.log logs/
```

### Step 3: 更新报告

评测完成后需要更新的内容：
1. **report.tex §7.1** — 补充 Qwen2.5 的评测结果表
2. **report.md §7.1** — 同步更新
3. **report.tex 摘要** — 补充跨架构对比结论
4. **figures/** — 重新生成包含 Qwen 数据的图表

### Step 4: 最终报告完善

- [ ] 补充 Qwen2.5 评测数据
- [ ] 更新 Loss 曲线图（含完整 Qwen 数据）
- [ ] 撰写跨架构分析讨论
- [ ] 编译 PDF 最终版

---

## 五、结果文件清单

```
results/
├── 主实验 (Llama-2)
│   ├── arc_{our,official,llama2}.json          # ✅
│   ├── triviaqa_{our,official,llama2}.json     # ✅
│   ├── popqa_nr_{our,official}.json            # ✅
│   └── popqa_ar_{our,official}.json            # ✅
│
├── I3 消融实验
│   ├── popqa_ar_our_no_grd.json                # ✅ 46.25%
│   ├── popqa_ar_our_no_ut.json                 # ✅ 50.54%
│   └── popqa_ar_our_no_score.json              # ✅ 46.32%
│
└── I1 Qwen2.5 评测 (🔄 进行中)
    ├── popqa_nr_qwen.json                      # 🔄
    ├── popqa_ar_qwen.json                      # 🔄
    ├── arc_qwen.json                           # 🔄
    └── triviaqa_qwen.json                      # 🔄
```

---

## 六、日志文件索引

| 日志 | 内容 | 状态 |
|------|------|:---:|
| `gen_train_20260423_010758.log` | Llama-2 Generator 训练 | ✅ |
| `train_qwen_20260426_072238.log` | Qwen2.5 训练 (原始, 79%) | ✅ |
| `train_qwen_resume_20260428_202428.log` | Qwen2.5 训练 (恢复, 完成) | ✅ |
| `eval_qwen_*.log` | Qwen2.5 评测 | 🔄 |
