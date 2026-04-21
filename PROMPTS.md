# Self-RAG 项目提示词手册

> 按照 `PROJECT_WORKFLOW.md` 的阶段顺序，在新窗口中依次使用以下提示词。
> 每个提示词对应一个独立的工作阶段，可在新窗口中粘贴使用。

---

## 硬件配置与运行环境说明

### 可用硬件

| 设备 | GPU | 显存 | 连接方式 | 适合任务 |
|------|-----|------|---------|---------|
| **实验室集群** | 8×A100 80GB + 16×A40 48GB | 640GB + 768GB | SSH 远程 | **所有训练、推理、评测** |
| 本地台式机 | RTX 5060 | 8GB | 本地 | 仅数据预处理、可视化、报告撰写 |

### 各步骤显存与时间预估

| 步骤 | 任务 | 最低显存需求 | 推荐配置 | 预计时间 |
|------|------|:----------:|---------|:--------:|
| Critic 训练 (7B) | 全量微调 + FSDP | ~35GB/卡 ×2 | 2×A100 80GB | 8-12 小时 |
| Generator 数据生成 | Critic 推理标注 150K 条 | ~16GB/卡 | 4×A40 48GB | 1-2 天 |
| Generator 训练 (7B) | 全量微调 + DeepSpeed ZeRO-3 | ~25GB/卡 ×4 | 4-8×A100 80GB | 1-2 天 |
| Generator 训练 (13B) | 全量微调 + DeepSpeed ZeRO-3 | ~40GB/卡 ×4 | 4×A100 80GB | 2-3 天 |
| 推理评测 (7B, vllm) | 模型加载 + 生成 | ~16GB | 1-2×A100 或 A40 | 4-6 小时/任务 |
| 推理评测 (13B, vllm) | 模型加载 + 生成 | ~28GB | 1×A100 80GB | 6-8 小时/任务 |
| 改进：Llama 3 8B Critic | 全量微调 + FSDP | ~38GB/卡 ×2 | 2×A100 80GB | 8-12 小时 |
| 改进：Llama 3 8B Generator | 全量微调 + DeepSpeed ZeRO-3 | ~28GB/卡 ×4 | 4-8×A100 80GB | 1-2 天 |
| 中文索引构建 (BGE) | Embedding 编码 + FAISS | ~4GB | 1×A40 48GB | 2-4 小时 |
| Demo 系统 (7B, vllm) | 在线推理服务 | ~16GB | 1×A40 48GB | 持续运行 |

> **关键结论**：RTX 5060 (8GB) **无法运行任何 7B 模型的训练或 fp16 推理**。即使使用 4-bit 量化加载 7B 模型（~4.5GB），加上 KV cache 和运行时开销也会接近 8GB 上限，极不稳定。**所有 GPU 任务必须在集群上通过 SSH 执行。**

### 推荐方案：全部任务放在集群执行（方案一）

本地台式机仅用于：SSH 连接集群、编辑代码、浏览结果文件、撰写报告。

---

## 提示词 0：集群连接与项目初始化

```
我正在做 NLP 课程大作业，选题为 Self-RAG（论文编号 14.4）。

我的工作环境：
- 实验室集群通过 SSH 连接，配有 8×A100 80GB 和 16×A40 48GB
- 本地只有 RTX 5060 8GB，不足以运行 7B 模型，所有 GPU 任务在集群上执行
- 集群上的项目目录为：[请填写集群上的实际路径，例如 /home/username/selfrag/]

请帮我完成以下初始化工作：
1. 在集群上创建项目目录结构：
   selfrag/
   ├── self-rag/       # 官方代码（已从 GitHub 克隆）
   ├── models/         # 存放模型权重
   ├── data/           # 存放训练/评测数据
   ├── outputs/        # 训练输出
   ├── results/        # 评测结果
   ├── figures/        # 可视化图表
   ├── scripts/        # 自定义脚本
   ├── demo/           # Demo 系统
   └── report/         # 课程报告

2. 克隆官方仓库：git clone https://github.com/AkariAsai/self-rag.git self-rag/
3. 确认集群上的 GPU 可用性：nvidia-smi
4. 确认 CUDA 版本和可用的 conda/module 系统

请注意：所有路径都用集群上的 Linux 路径，所有命令都通过 SSH 在集群上执行。
```

---

## 提示词 1：环境部署与依赖安装

```
集群项目目录：[填写集群路径，如 /home/username/selfrag/]
官方代码目录：self-rag/

请在集群上帮我完成以下环境部署工作：
1. 阅读 self-rag/ 目录下的 requirements.txt 和 environment.yml，了解所有依赖
2. 创建一个 conda 环境（名称 selfrag），安装所有必要依赖（vllm, transformers, deepspeed, torch, faiss-gpu 等）
3. 用 nvidia-smi 确认 GPU 可用，用 python -c "import torch; print(torch.cuda.device_count())" 确认 PyTorch 能看到所有 GPU
4. 验证 vllm 能正常导入：python -c "from vllm import LLM; print('vllm OK')"
5. 验证 deepspeed 安装：ds_report

请在安装前先列出完整的依赖清单，确认无冲突后再执行安装。
注意：集群环境是 Linux，请使用 Linux 命令。
```

---

## 提示词 2：模型下载（预训练 Self-RAG + 基座模型）

```
集群项目目录：[填写集群路径]
conda 环境：selfrag

请在集群上帮我下载以下模型到 models/ 目录：

1. 官方预训练 Self-RAG 模型（用于推理验证和基线对比）：
   - selfrag/selfrag_llama2_7b（~14GB）
   - selfrag/selfrag_llama2_13b（~26GB，如存储空间允许）

2. 基座模型（用于从头训练复现）：
   - meta-llama/Llama-2-7b-hf（~14GB）

3. 改进实验用的基座模型（后续阶段使用）：
   - meta-llama/Llama-3.1-8B（~16GB）
   - Qwen/Qwen2.5-7B（~14GB）

请使用 huggingface-cli download 或 Python 脚本下载。如果集群需要配置代理才能访问 HuggingFace，请帮我设置。如果需要 HuggingFace token 登录获取 Llama 系列模型，请提示我。

下载完成后验证每个模型能正常加载（在单 GPU 上 python -c 快速测试）。

提示：如果下载速度慢，可尝试使用 HuggingFace 镜像站 https://hf-mirror.com。
```

---

## 提示词 3：数据下载与准备

```
集群项目目录：[填写集群路径]
conda 环境：selfrag

请在集群上帮我下载并整理 Self-RAG 所需的全部训练数据和评测数据：

**训练数据：**
1. Critic 训练数据（GPT-4 标注的反思 token 数据）
   - 官方链接：https://drive.google.com/file/d/1IN1XcIOYtRIGWITJ4LKRgfITT-uUwk_W/view
2. Generator 训练数据（150K 实例）
   - HuggingFace: selfrag/selfrag_train_data
   - 或 Google Drive: https://drive.google.com/file/d/10G_FozUV4u27EX0NjwVe-3YMUMeTwuLk/view

**评测数据：**
3. 评测数据集（PubHealth, ARC-Challenge, TriviaQA, PopQA, ASQA, FactScore）
   - 官方链接：https://drive.google.com/file/d/1TLKhWjez63H4uBtgCxyoyJsZi-IMgnDb/view

请将数据保存到 data/ 子目录下，按训练/评测分类组织。

注意：集群可能无法直接访问 Google Drive。如有困难，请提供 gdown 命令或建议我先在本地下载再 scp 传到集群。

下载完成后检查数据格式，列出每个数据集的样本数量和字段结构。
```

---

## 提示词 4：快速推理验证（使用官方预训练模型）

```
集群项目目录：[填写集群路径]
conda 环境：selfrag
预训练模型路径：models/selfrag_llama2_7b
硬件：使用 1×A40 48GB 即可（推理 7B fp16 约需 16GB 显存）

请帮我用官方预训练的 Self-RAG 模型进行推理验证：

1. 使用 vllm 加载 selfrag_llama2_7b 模型
2. 测试以下场景，观察反思 Token 的行为：
   a. 简单事实问题（不需要检索）："What is 2+2?"
   b. 需要检索的知识问题："Can you tell me the difference between llamas and alpacas?"（附带一段检索到的段落）
   c. 中文问题测试："中国的首都是哪里？"
3. 打印完整输出，包括所有反思 Token（[Retrieve], [No Retrieval], [Relevant], [Irrelevant], [Fully supported], [Partially supported], [No support], [Utility:1-5]）
4. 分析反思 Token 的分布规律

请参考 self-rag/README.md 中 Quick Start 部分的代码格式。将验证脚本保存为 scripts/quick_inference.py。

提示：如果需要在集群上提交 GPU 作业（如 SLURM），请帮我编写对应的作业提交脚本。
```

---

## 提示词 5：Critic 模型训练

```
集群项目目录：[填写集群路径]
conda 环境：selfrag
基座模型：models/Llama-2-7b-hf
Critic 训练数据：data/critic/（前面已下载的 GPT-4 标注数据）
硬件：2×A100 80GB（FSDP 全量微调，预计每卡占用 ~35GB，训练约 8-12 小时）

请帮我完成 Critic 模型训练：

1. 阅读 self-rag/data_creation/ 目录下的训练脚本（train_special_tokens.py），理解：
   - 如何在 tokenizer 中注册 4 种反思 Token
   - 训练数据的格式和加载方式
   - FSDP 并行训练的配置

2. 编写并运行 Critic 训练命令：
   - 基座模型：Llama-2-7b-hf
   - 使用 2×A100 + FSDP full_shard
   - 训练 3 个 epoch，lr=2e-5，batch_size=1，grad_accum=8
   - 保存到 outputs/critic_llama2_7b/

3. 训练完成后：
   - 绘制 loss 曲线
   - 在 held-out 数据上评测 Critic 的反思 Token 预测准确率
   - 保存训练日志和评测结果

请先检查训练脚本，确认参数无误后再开始训练。将训练启动脚本保存为 scripts/train_critic.sh。

建议使用 nohup 或 tmux/screen 在后台运行训练，防止 SSH 断连导致训练中断：
nohup bash scripts/train_critic.sh > logs/critic_train.log 2>&1 &
```

---

## 提示词 6：Generator 训练数据生成（可选，如使用官方数据可跳过）

```
集群项目目录：[填写集群路径]
conda 环境：selfrag
训练好的 Critic 模型：outputs/critic_llama2_7b/
硬件：4×A40 48GB（Critic 推理标注，预计每卡 ~16GB，耗时 1-2 天）

如果我们要自己生成 Generator 训练数据（而非直接使用官方提供的 150K 数据），请帮我完成以下工作：

1. 阅读 self-rag/data_creation/generator/ 目录下的 README.md 和相关脚本
2. 理解 Generator 训练数据生成流程：
   - 原始语料 → Retriever 检索相关段落 → Critic 标注反思 Token → 组装训练样本
3. 使用训练好的 Critic 模型和 Contriever 检索器，为原始语料生成带反思标注的训练数据
4. 对比自己生成的数据与官方数据的分布差异

如果生成过程过于耗时或复杂，请建议我们直接使用官方提供的 150K 训练数据，并解释原因。

注意：使用 tmux/screen 在后台运行，避免 SSH 断连。
```

---

## 提示词 7：Generator 模型训练

```
集群项目目录：[填写集群路径]
conda 环境：selfrag
基座模型：models/Llama-2-7b-hf
Generator 训练数据：data/generator/（150K 实例）
硬件：4-8×A100 80GB（DeepSpeed ZeRO-3 全量微调，预计每卡 ~25GB，训练 1-2 天）

请帮我完成 Generator 模型训练：

1. 阅读 self-rag/retrieval_lm/ 目录下的训练脚本（script_finetune_7b.sh），理解：
   - DeepSpeed 配置（ZeRO Stage 3）
   - 训练超参数设置
   - 数据加载和预处理方式

2. 配置并运行 Generator 训练：
   - 基座模型：Llama-2-7b-hf
   - 使用 4-8×A100 + DeepSpeed ZeRO-3
   - 训练数据：150K 实例
   - 保存到 outputs/generator_llama2_7b/

3. 训练完成后：
   - 绘制 loss 曲线
   - 用几个测试 query 快速验证输出是否包含正确的反思 Token
   - 保存训练日志

请确保 DeepSpeed 配置文件正确适配集群的 A100 80GB。
使用 tmux/screen + nohup 在后台运行，避免 SSH 断连。

提示：官方说 7B 训练用了 8×A100 40GB；我们的 A100 是 80GB，显存更充裕，可以尝试增大 batch_size 来加速训练。
```

---

## 提示词 8：全面评测与基线对比

```
集群项目目录：[填写集群路径]
conda 环境：selfrag
我们复现训练的模型：outputs/generator_llama2_7b/
官方预训练模型：models/selfrag_llama2_7b（用于交叉验证）
评测数据：data/eval/
硬件：评测可以同时利用 A100 和 A40
  - Self-RAG 模型评测：2×A100（vllm 推理，每卡 ~16GB）
  - 基线模型评测（Vanilla Llama 2 等）：可并行用 A40

请帮我完成全面评测与基线对比实验：

**评测我们复现的 Self-RAG 模型：**
1. Short-form 任务：PubHealth (Accuracy), ARC-Challenge (Accuracy), TriviaQA (EM), PopQA (EM)
2. Long-form 任务：ASQA (Correctness + Citation Precision/Recall), FactScore

**评测基线模型（对比组）：**
3. Vanilla Llama 2 7B（无检索，直接生成）
4. Standard RAG（始终检索 Top-5 + 拼接生成）
5. 官方预训练 Self-RAG 模型（交叉验证我们的复现质量）

**输出要求：**
6. 生成一张完整的对比表格（Markdown 格式），包含所有模型在所有任务上的性能
7. 计算我们复现的结果与论文原始报告数字的差异
8. 将所有评测脚本整理到 scripts/eval/ 目录下
9. 将结果保存为 results/reproduction_results.json

请参考 self-rag/README.md 的 Inference 部分设计评测脚本。每个评测任务分开运行，便于追踪问题。

提示：可以写一个 run_all_eval.sh 脚本，按顺序提交所有评测任务到后台，然后用 tail -f 监控进度。
```

---

## 提示词 9：改进实验 I1 — 基座模型升级（Llama 3）

```
集群项目目录：[填写集群路径]
conda 环境：selfrag
Llama 3 基座模型：models/Llama-3.1-8B
硬件：
  - Critic 训练：2×A100 80GB（每卡 ~38GB，8-12 小时）
  - Generator 训练：4-8×A100 80GB（每卡 ~28GB，1-2 天）

这是我们的核心改进实验。请帮我将 Self-RAG 的训练从 Llama 2 升级到 Llama 3：

1. **分析兼容性**：对比 Llama 2 和 Llama 3 的 tokenizer 差异，确认反思 Token 的注册方式是否需要修改

2. **Critic 训练（Llama 3 基座）**：
   - 使用 meta-llama/Llama-3.1-8B 替换 Llama-2-7b-hf
   - 使用相同的 Critic 训练数据和超参数
   - 保存到 outputs/critic_llama3_8b/

3. **Generator 训练（Llama 3 基座）**：
   - 使用 meta-llama/Llama-3.1-8B 替换 Llama-2-7b-hf
   - 使用相同的 Generator 训练数据
   - 保存到 outputs/generator_llama3_8b/

4. **全量评测**：
   - 在全部 6 个评测任务上评测 Llama 3 版 Self-RAG
   - 与 Llama 2 版结果直接对比

5. **结果分析**：
   - 生成 Llama 2 vs Llama 3 的对比表格
   - 分析升级基座模型带来的性能变化
   - 保存到 results/llama3_upgrade_results.json

如果遇到兼容性问题（如 tokenizer 差异、DeepSpeed 配置变化），请详细说明并给出解决方案。
```

---

## 提示词 10：改进实验 I2 — 中文数据集适配

```
集群项目目录：[填写集群路径]
conda 环境：selfrag
硬件：
  - 中文索引构建（BGE 编码）：1×A40 48GB（~4GB 显存，2-4 小时）
  - 中文评测推理：1-2×A40 48GB

请帮我完成 Self-RAG 的中文数据集适配实验：

1. **中文检索索引构建**：
   - 下载中文维基百科段落数据（或使用 DuReader 的文档集合）
   - 使用 BAAI/bge-base-zh-v1.5 或类似中文 embedding 模型编码段落
   - 构建 FAISS 索引，保存到 data/chinese/wiki_zh_index/

2. **中文 QA 评测数据准备**：
   - 下载并格式化中文 QA 数据集（推荐 WebQA 或 DuReader 的子集）
   - 格式与英文评测数据保持一致
   - 保存到 data/chinese/eval/

3. **中文评测实验**：
   - 使用 Qwen 2.5 7B 作为基座（因为其中文能力远强于 Llama）
   - 对比三种方法在中文 QA 上的表现：
     a. Vanilla Qwen 2.5（无检索）
     b. Standard RAG + Qwen 2.5（始终检索）
     c. Self-RAG + Qwen 2.5（如果已在 Qwen 上训练过；否则用 Llama 3 版 Self-RAG 测试）

4. **结果分析**：
   - 分析 Self-RAG 在中文场景下的表现
   - 与英文结果对比，讨论跨语言泛化性
   - 保存到 results/chinese_results.json

注意：如果在 Qwen 上完整训练 Self-RAG 时间不够，可以只做推理测试（使用已训练好的 Llama 3 版 Self-RAG 模型），重点分析框架在中文上的 zero-shot 表现。
```

---

## 提示词 11：消融实验

```
集群项目目录：[填写集群路径]
conda 环境：selfrag
复现的 Self-RAG 模型：outputs/generator_llama2_7b/
硬件：推理评测为主，每个消融用 1-2×A40 48GB 即可，可多组并行

请帮我完成消融实验，验证 Self-RAG 各组件的贡献：

**消融实验设计：**

| ID | 设置 | 目的 |
|----|------|------|
| Ab1 | 移除所有反思 Token → 退化为标准 RAG | 验证 Self-RAG 框架的核心价值 |
| Ab4 | 始终检索（移除 [No Retrieval] 选项） | 验证自适应检索 vs 强制检索 |
| Ab5 | 不同 Top-K 值对比（K=1,3,5,10,20） | 检索数量对性能的影响 |
| Ab6 | Llama 2 7B vs Llama 3 8B vs Qwen 2.5 7B 基座对比 | 框架的模型无关性（复用改进实验 I1 结果） |

**评测范围**：每个消融至少在 PubHealth + TriviaQA + ASQA 三个任务上评测

**输出要求**：
1. 生成消融结果汇总表格（Markdown 格式）
2. 绘制关键对比图表（柱状图/折线图）
3. 编写每组消融的分析说明（2-3 句话）
4. 将结果保存为 results/ablation_results.json
5. 将所有消融评测脚本整理到 scripts/ablation/ 目录下

提示：消融实验主要是推理评测，显存需求较低（~16GB），可以充分利用 A40 集群并行跑多组实验，大幅缩短总时间。
```

---

## 提示词 12：可视化分析与 Case Study

```
集群项目目录：[填写集群路径]
conda 环境：selfrag
所有实验结果：results/ 目录下的 JSON 文件

请帮我完成可视化分析和案例研究：

**可视化图表（使用 matplotlib + seaborn，统一风格，300dpi）：**

1. **检索触发率柱状图**：各评测任务中 [Retrieval] vs [No Retrieval] 的比例
2. **反思 Token 分布热力图**：[IsRel], [IsSup], [IsUse] 各类别在不同任务上的分布
3. **基座模型对比雷达图**：Llama 2 vs Llama 3 vs Qwen 在各任务上的多维对比
4. **消融实验柱状图**：各消融设置 vs Full Self-RAG 的性能对比
5. **Top-K 敏感性折线图**：K=1,3,5,10,20 对应的性能变化曲线
6. **复现 vs 论文原始结果对比图**：我们的复现结果与论文数字的差异

**Case Study（3-5 个精选案例）：**
请从评测结果中挑选以下类型的典型案例：
- ① Self-RAG 正确判断不需检索的简单问题
- ② Self-RAG 正确拒绝了不相关的检索结果
- ③ Self-RAG 通过检索修正了 Vanilla LLM 的事实性错误
- ④ Self-RAG 的失败案例（检索了但仍然错误）
- ⑤ 中文场景下的一个典型案例

每个案例包含：输入问题、检索段落（如有）、各反思 Token 的输出、最终回答、正确答案、分析说明。

**将所有图表保存到 figures/ 目录，Case Study 文本保存到 results/case_study.md。**

注意：可视化绑脚本可以在集群上无 GUI 运行（matplotlib 使用 Agg backend）：
import matplotlib; matplotlib.use('Agg')
```

---

## 提示词 13：Demo 系统搭建

```
集群项目目录：[填写集群路径]
conda 环境：selfrag
Self-RAG 模型：outputs/generator_llama2_7b/（或官方预训练模型）
硬件：1×A40 48GB 即可运行 7B 推理服务（~16GB 显存）

请帮我在集群上构建一个交互式 Demo 系统，用于报告展示：

**功能需求：**
1. 用户输入一个问题（支持中/英文）
2. 系统展示 Self-RAG 的完整推理过程：
   - Step 1: 模型判断是否需要检索（显示 [Retrieval] 或 [No Retrieval]）
   - Step 2: 如果需要检索，展示检索到的 Top-K 段落
   - Step 3: 展示各反思 Token 的预测（[IsRel], [IsSup], [IsUse] 的具体值）
   - Step 4: 展示最终生成的回答
3. 侧边栏显示模型配置（基座模型、Top-K 值、温度等）
4. 支持对比模式：同时展示 Vanilla LLM 和 Self-RAG 的输出

**技术选型：**
- 使用 Gradio 构建 Web UI
- 后端使用 vllm 加载模型

**部署方式：**
- 在集群上启动 Gradio 服务，通过 SSH 端口转发在本地浏览器访问：
  ssh -L 7860:localhost:7860 username@cluster_ip
- 或使用 Gradio 的 share=True 生成公网链接

**输出：**
- 将 Demo 代码保存为 demo/app.py
- 提供启动脚本 demo/run_demo.sh
- 截取几张 Demo 运行截图保存到 figures/demo/
```

---

## 提示词 14：课程报告撰写

```
项目目录：[填写集群路径]（实验结果在集群上）
本地路径：d:\USTC\2026Spring\自然语言处理\14.4 Self-RAG\（报告可在本地撰写）
所有实验结果：results/ 目录
所有图表：figures/ 目录
Case Study：results/case_study.md
PROJECT_WORKFLOW.md：d:\USTC\2026Spring\自然语言处理\PROJECT_WORKFLOW.md

请帮我撰写 NLP 课程大作业的阅读报告。

**格式要求：**
- 中文撰写，篇幅 5000 字以上
- 输出为 Markdown 格式，保存到 report/report.md

**报告结构（严格遵循课程要求）：**

§1 背景介绍（~800字）
- RAG 技术的发展脉络：从传统 QA → 检索增强 → Self-RAG 的演进
- Self-RAG 解决的核心问题：何时检索、如何评估检索质量、如何自我批判

§2 现有方法及其局限性（~600字）
- Vanilla LLM 的局限（幻觉问题）
- Standard RAG 的局限（始终检索的冗余和噪声）
- Self-RAG 的优越性（自适应检索 + 自我反思）

§3 论文方法详解（~1200字）
- 4 种反思 Token 的定义和作用
- Critic 模型的训练方法
- Generator 模型的训练方法
- 推理时的 tree decoding 机制

§4 实验结果与分析（~1500字）
- 复现结果 vs 论文原始结果的对比分析
- 基线对比实验结果
- 消融实验结果与分析
- 改进实验结果：基座升级（Llama 3）+ 中文适配
- 引用 figures/ 下的图表

§5 案例展示（~600字）
- 插入 results/case_study.md 中的典型案例
- Demo 系统截图
- 分析 Self-RAG 的实际工作效果

§6 局限性与拓展方向（~500字）
- 当前局限：Critic 训练依赖 GPT-4、检索器质量瓶颈、推理开销
- 可能的拓展：多模态 Self-RAG、更高效的反思机制、与 DPO 结合

**附录**：超参数表、完整实验数据、环境配置、组员分工说明

请在撰写时：
1. 使用专业的学术语言
2. 每个实验结果配合对应的图表引用
3. 在报告结尾注明两位组员的分工
4. 确保总字数超过 5000 字
```

---

## 使用顺序建议

```
阶段一（W1-W2 基建）：提示词 0 → 1 → 2 → 3 → 4
阶段二（W3-W5 复现）：提示词 5 → 6(可选) → 7 → 8
阶段三（W5-W7 改进）：提示词 9 → 10 → 11 → 12 → 13
阶段四（W7-W8 报告）：提示词 14
```

> **提示**：每个提示词在新窗口中使用时，先粘贴提示词，再将 `[填写集群路径]` 替换为实际路径（如 `/home/username/selfrag/`），然后根据前序步骤的实际结果追加补充信息。如果某一步遇到错误，可以在同一窗口中继续调试。
>
> **SSH 最佳实践**：所有长时间运行的训练任务，务必使用 `tmux` 或 `screen` 在集群上创建持久会话，防止 SSH 断连导致训练中断。
