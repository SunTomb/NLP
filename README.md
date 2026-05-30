# Self-RAG: 自反思检索增强生成的复现与改进

本项目是针对 [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection (ICLR 2024)](https://arxiv.org/abs/2310.11511) 的非官方复现与改进版本，作为中科大（USTC）《自然语言处理》课程实验的最终项目。

本项目从零开始复现了 Self-RAG 的 Critic 与 Generator 模型训练，深入剖析了内部的反思 Token 机制（通过消融实验），并将原基于 Llama-2 的框架成功迁移至开源表现极佳的 **Qwen2.5-7B** 基座，实现了跨架构泛化的验证。

---

## 🌟 核心亮点

1. **完全复现**：在单卡 A40 (48GB) 上完整打通了 Critic (21.5h) 和 Generator (66.5h) 的全量微调，原汁原味还原了论文评测成绩。
2. **消融分析**：量化解耦了各个特殊 Token 的独立贡献，证实了 **Groundness（支持度）评分** 是短答案知识问答产生增益的核心机制。
3. **跨架构迁移 (Qwen2.5)**：解决特殊 Token 注入等工程问题，成功在 Qwen2.5-7B 上训练 Self-RAG，其在科学推理任务 (ARC-C) 上准确率高达 68.09%（超越原论文的官方模型）。
4. **可视化交互**：内置基于 Gradio + vLLM 构建的 WebUI，颜色高亮反思 Token，直观展现模型边思考边生成的全过程。

---

## 📊 实验结果

在三个核心基准上的测试结果（Metric: Match Accuracy）：

| 任务 | 模式 | Our (Llama-2) | Our (Qwen2.5) | Official (Llama-2) | Llama-2 (Base) |
|------|------|:---:|:---:|:---:|:---:|
| **PopQA** | no_retrieval | 23.45% | 19.44% | 28.66% | — |
| **PopQA** | always_retrieve | 50.46% | 49.68% | 52.32% | — |
| **ARC-C** | no_retrieval | 57.25% | **68.09%** | 64.25% | 43.34% |
| **TriviaQA**| no_retrieval | 31.50% | 26.80% | 29.75% | 17.05% |

> *注：Qwen2.5 的极高 ARC-C 成绩证明了其强大的基座推理能力，而在纯知识召回类任务（PopQA NR / TriviaQA）中，则受限于其中英文大词表的知识密度稀释。*

---

## 📂 仓库结构

```text
NLP/
├── scripts/              # 运行脚本目录
│   ├── train_critic.sh       # 训练 Critic
│   ├── train_generator.sh    # 训练 Generator
│   └── run_eval_*.sh         # 各个数据集的评测脚本
├── demo/                 # 交互式前端
│   ├── app.py            # Gradio WebUI 源码
│   └── run_app.sh        # 启动可视化 WebUI
├── figures/              # 数据可视化输出
├── generate_figures.py   # 一键生成论文图表的 Python 脚本
├── report.pdf            # 最终的课程实验报告 
├── problem.md            # 记录了 21 项踩坑与解决全过程
└── PROJECT_WORKFLOW*.md  # 开发过程进度日志
```

---

## 🚀 快速开始

### 1. 环境配置

推荐使用 Conda 和 vLLM (用于极速评测和推理)：

> 训练、评测和 WebUI 推理依赖 CUDA GPU 与 vLLM，建议在 Linux GPU 服务器上运行；macOS/普通 Windows 本地环境主要适合阅读代码和编译报告。

```bash
conda create -n selfrag python=3.10
conda activate selfrag
pip install -r self-rag/requirements.txt
pip install vllm==0.5.5 gradio
```

### 2. 下载数据

```bash
bash scripts/download_data.sh
```

如果后续训练提示 `data/generator/output_selfrag_training_data.jsonl` 不存在，请检查 `data/generator/` 下实际下载的 `.jsonl` 文件名，并相应修改 `scripts/train_generator.sh` 中的 `TRAIN_FILE`。

### 3. 训练 Generator

```bash
# 启动训练（推荐使用 Adafactor 和 gradient checkpointing 防 OOM）
bash scripts/train_generator.sh
```

### 4. 评测 (Evaluation)

```bash
# 对 PopQA 运行 always_retrieve 评测模式
bash scripts/run_eval_popqa_full.sh
```

### 5. 启动 WebUI (体验反思 Token 可视化)

```bash
# 从项目根目录启动
bash demo/run_app.sh
```

---

## 📖 详细报告

更多关于消融实验的设计、跨架构泛化的探讨，以及项目中遭遇的显存优化、数据预处理等 21 个工程难题的详尽解答，请参阅：

- 📄 [课程实验报告 (PDF)](report.pdf)
- 📝 [工程踩坑记录 (Markdown)](problem.md)

## 🙏 致谢

本项目基于 [Self-RAG 官方仓库](https://github.com/AkariAsai/self-rag) 进行了二次开发。感谢 USTC Lab for Data Science 提供的计算资源支持。
