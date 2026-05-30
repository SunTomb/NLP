# 自然语言处理课程报告

## Self-RAG: 自反思检索增强生成的复现与改进

**姓名**：叶盛豪 &emsp; **学号**：PB24000227  
**姓名**：闪瑜 &emsp; **学号**：PB23051016  
**日期**：2026 年 4 月 29 日

---

## 摘要

检索增强生成（Retrieval-Augmented Generation, RAG）是提升大语言模型事实准确性的重要范式，但传统 RAG 方法存在不加区分地检索外部文档、无法判断检索结果质量等问题。Self-RAG 通过引入反思 Token（Reflection Tokens）机制，使语言模型能够在推理过程中自主决定是否检索、评估检索段落的相关性与支持度，并对生成结果进行自我批判。

本报告对 Self-RAG 进行了系统性的复现与改进实验。在复现阶段，我们基于 Llama-2-7B 成功训练了 Critic 和 Generator 模型，在 PopQA、ARC-Challenge 和 TriviaQA 三个数据集上的评测结果与原论文保持一致。在改进阶段，我们完成了三项工作：（1）消融实验揭示了 Groundness 评分机制是检索增强的核心贡献因子（贡献 4.22 个百分点），而 Utility 评分对短答案 QA 任务影响甚微；（2）将 Self-RAG 框架迁移至 Qwen2.5-7B 基座，验证了方法的跨架构泛化能力；（3）构建了基于 Gradio 的交互式 Demo 系统。此外，我们详细记录了 21 项工程问题及其解决方案，并开源了完整代码与报告材料，项目仓库地址为：https://github.com/SunTomb/NLP。本报告由叶盛豪、闪瑜合作完成，二人工作量各占 50%。

**关键词**：检索增强生成；反思 Token；Self-RAG；消融实验；大语言模型

---

## 1. 引言

大语言模型（Large Language Models, LLMs）在自然语言处理的众多任务上取得了显著进展，但其核心局限之一在于**幻觉**（Hallucination）——模型可能生成看似流畅但事实错误的内容。这一问题在知识密集型任务（如开放域问答）中尤为突出，因为模型的参数化知识不可避免地存在时效性和覆盖范围的限制。

从 NLP 任务角度看，本文关注的是**知识密集型文本生成**，尤其是开放域问答、科学推理问答和事实型短答案生成。此类任务要求模型不仅能够理解用户问题，还需要从大规模背景知识中定位相关事实，并将证据转化为简洁、准确、可验证的自然语言回答。与情感分类、命名实体识别等传统判别式任务相比，知识密集型生成同时考验模型的语言组织能力、事实记忆能力和证据利用能力；与普通聊天生成相比，它对答案真实性和证据一致性的要求更高。PopQA 中的长尾实体问题、TriviaQA 中的开放域事实问答，以及 ARC-Challenge 中的科学常识推理，分别代表了事实召回、知识覆盖和推理应用三类典型挑战。

检索增强生成（RAG）通过在推理时引入外部检索器来缓解这一问题：模型先从知识库中检索相关文档，再基于检索结果生成回答。然而，传统 RAG 方法存在以下不足：

1. **无差别检索**：对所有输入查询一律执行检索，即使模型内部已拥有足够的知识来回答问题，也会引入不必要的计算开销和检索噪声；
2. **缺乏质量判断**：模型无法评估检索到的段落是否真正支持所生成的回答，可能导致"检索了但没用对"的困境；
3. **端到端优化困难**：检索器和生成器通常是独立训练的，二者之间缺乏有效的反馈机制。

Asai 等人 (2024) 提出的 **Self-RAG**（Self-Reflective Retrieval-Augmented Generation）从根本上改变了 RAG 的范式。其核心创新在于引入了一组**反思 Token**（Reflection Tokens），使语言模型在生成过程中能够：（1）自主判断当前是否需要检索；（2）评估检索段落的相关性；（3）判断生成内容是否被检索证据支持；（4）对最终输出的整体质量进行评分。

**本报告的主要贡献**：
- 在单卡 NVIDIA A40 (48GB) 上完整复现了 Self-RAG 的 Critic 和 Generator 训练流程
- 通过消融实验定量分析了各反思 Token 评分机制的独立贡献
- 将 Self-RAG 框架成功迁移至 Qwen2.5-7B 基座模型
- 构建了基于 Gradio + vLLM 的交互式 Demo 系统
- 系统记录了 21 项工程实践问题及解决方案

---

## 2. 相关工作

**检索增强生成（RAG）。** Lewis 等人 (2020) 提出的 RAG 模型将预训练语言模型与稀疏/稠密检索器相结合，在开放域 QA 上取得了显著提升。后续工作如 REALM (Guu et al., 2020)、Atlas (Izacard et al., 2023) 进一步优化了检索器与生成器的联合训练。这类方法的共同优点是能够将模型参数之外的知识引入生成过程，从而缓解知识过时、长尾事实缺失和事实幻觉等问题；但其局限也很明显：检索通常作为固定前置步骤，对所有问题统一执行，既增加了推理成本，也可能把无关或矛盾证据引入上下文。

**现有方法的局限性。** 传统 RAG 系统通常假设“检索到的文档越多越好”，但实际开放域问答和知识密集型生成并不总是需要外部知识。例如简单常识题、数学题或模型已掌握的高频事实，强制检索反而可能带来噪声；而对于长尾实体、时效性知识或证据依赖较强的问题，缺少检索又会导致模型依赖不可靠的参数化记忆。此外，许多 RAG 方法只关注“检索—生成”链路本身，并不显式判断检索段落是否相关、生成答案是否被证据支持、最终回答是否真正有用。因此，一旦检索器返回弱相关段落，生成器仍可能流畅地编造答案，形成“看似有引用、实则无支撑”的幻觉。

**Self-RAG 的优越性。** Self-RAG 的主要优势在于把检索决策、证据评估和生成质量判断都转化为语言模型可学习、可输出的反思 Token。相比固定检索策略，它能够在生成过程中显式表达是否需要检索；相比只拼接文档的 RAG，它进一步评估段落相关性和回答支持度；相比外部后处理打分器，它将自我批判能力内化进同一个生成模型中。因此，Self-RAG 不只是“带检索的生成模型”，而是一个具备自适应检索和自我评估能力的生成框架。

**自适应检索。** FLARE (Jiang et al., 2023) 通过检测低置信度 Token 来触发检索；Self-RAG 则通过训练模型生成显式的检索决策 Token，实现了更精细的自适应控制。

**自我批判与反思。** Constitutional AI (Bai et al., 2022) 和 Self-Refine (Madaan et al., 2023) 让模型对自身输出进行批判和迭代修正。Self-RAG 将这一思想内化为词表中的反思 Token，使模型在单次前向传播中即可完成生成与自我评估。

---

## 3. Self-RAG 方法概述

Self-RAG 的核心思想是将检索决策和质量评估内化为语言模型的生成过程。模型的词表在原始 Token 基础上扩展了一组反思 Token，模型在自回归生成中穿插生成这些特殊 Token，实现对自身行为的元认知控制。

### 3.1 反思 Token 机制

Self-RAG 定义了四类共 15 个反思 Token：

| 类别 | Token | 语义 |
|------|-------|------|
| 检索决策 | `[Retrieval]` / `[No Retrieval]` | 模型判断当前是否需要检索外部文档 |
| 相关性评估 | `[Relevant]` / `[Irrelevant]` | 检索到的段落是否与查询相关 |
| 支持度评估 | `[Fully supported]` / `[Partially supported]` / `[No support / Contradictory]` | 生成内容是否被检索证据支持（Groundness） |
| 效用评分 | `[Utility:1]` ~ `[Utility:5]` | 生成结果对原始查询的整体有用性 |

### 3.2 训练流程

**阶段一：Critic 模型训练。** 使用 GPT-4 对训练数据中的检索段落和生成文本进行多维度标注，得到约 48.5K 条标注数据。基于 Llama-2-7B 基座训练 Critic 模型，使其学会预测各类反思 Token。Critic 的作用不是直接回答用户问题，而是学习“什么时候需要检索”“某个段落是否相关”“生成内容是否被证据支持”等判断能力。换言之，它相当于为 Generator 提供一个自动化的反思标注器，使大量普通指令数据能够被转化为带有过程监督信号的训练样本。

$$\mathcal{L}_{\text{critic}} = -\sum_{t} \log P(y_t^{\text{reflect}} \mid x, y_{<t})$$

**阶段二：Generator 模型训练。** 使用训练好的 Critic 对大规模指令微调数据进行自动标注，得到约 145K 条带反思 Token 的训练数据。同样基于 Llama-2-7B，以因果语言模型目标训练 Generator。训练完成后，Generator 不仅学习普通自然语言回答，还学习在回答过程中输出反思 Token。这样做的关键意义在于：反思不再是外部模块对生成结果的事后检查，而是模型自回归生成序列的一部分，能够和正文内容共同参与解码与排序。

$$\mathcal{L}_{\text{gen}} = -\sum_{t} \log P(y_t \mid x, y_{<t}), \quad y_t \in \mathcal{V} \cup \mathcal{V}_{\text{reflect}}$$

在实现层面，反思 Token 被加入 tokenizer 的 additional special tokens 中，模型的 embedding 矩阵随之扩展。训练时这些特殊标记和普通文本 token 一样参与语言建模损失，因此模型可以通过最大似然学习其出现位置和语义功能。推理时，这些 token 又可以被解析为控制信号或打分信号，例如 `[Retrieval]` 表示模型认为需要外部知识，`[Fully supported]` 表示当前候选回答被证据充分支持，`[Utility:5]` 表示模型认为回答对用户问题高度有用。

### 3.3 推理流程

**No Retrieval 模式**：模型直接基于内部参数化知识生成回答。

**Always Retrieve 模式**：对每个查询，从预检索的段落集合中选取 top-k 段落（k=5），模型为每个段落独立生成候选回答，使用 Groundness 和 Utility 评分排序选出最优输出：

$$\text{score}(y \mid d) = w_g \cdot P(\texttt{[Fully supported]} \mid y, d) + w_u \cdot P(\texttt{[Utility:5]} \mid y)$$

---

## 4. 实验设置

### 4.1 硬件环境

| 配置项 | 详情 |
|--------|------|
| GPU | NVIDIA A40 (48GB VRAM) × 8（使用单卡） |
| CPU | Intel Xeon Gold 6330 (56 cores) |
| 内存 | 1TB DDR4 |
| PyTorch | 2.4.0 + CUDA 12.4 |
| 推理引擎 | vLLM 0.5.5 |
| Python | 3.10.20 (Conda 隔离环境) |

### 4.2 数据集

| 数据集 | 样本数 | 指标 | 任务描述 |
|--------|:---:|------|---------|
| PopQA | 1,399 | Accuracy (match) | 长尾实体知识 QA，每条含 25 个预检索段落 |
| ARC-Challenge | 1,172 | Accuracy (match) | 科学推理选择题（四选一） |
| TriviaQA | 2,000 | Accuracy (match) | 开放域知识问答（子集采样） |

### 4.3 训练配置

| 参数 | Critic | Generator |
|------|:---:|:---:|
| 基座模型 | Llama-2-7B | Llama-2-7B |
| 训练数据量 | 48.5K | 145K |
| Epoch 数 | 3 | 3 |
| 有效 Batch Size | 16 | 16 |
| 优化器 | Adafactor | Adafactor |
| 学习率 | 2×10⁻⁵ | 2×10⁻⁵ |
| 最大序列长度 | 2048 | 2048 |
| 精度 | BF16 | BF16 |
| 训练时间 | ~21.5h | ~66.5h |
| 显存占用 | ~40GB | ~44GB |

### 4.4 评测指标与基线

**评测指标**：所有任务统一使用 **match accuracy**——将模型生成文本与参考答案列表进行子串匹配，命中任一参考答案即视为正确。采用这一指标的原因是三个评测任务都以短答案或选项答案为主，参考答案通常包含多个别名或等价表达，严格字符串完全匹配会低估模型性能。与此同时，我们也在工程调试中发现，子串匹配对长文本输出并不稳健，因此在 PopQA 等任务上必须控制生成长度并使用包含检索段落的公平评测设置。

**基线模型**：Our (Llama-2)（本实验从零训练）、Official（原论文权重）、Llama-2-7B（未微调基座）。这三个基线分别对应“自主复现模型”“官方发布模型”和“未经 Self-RAG 微调的原始大模型”，能够从复现一致性和方法有效性两个角度评价实验结果。

---

## 5. 复现实验

### 5.1 Critic 模型训练

Critic 模型在 48.5K 标注数据上训练 3 个 epoch，总耗时约 21.5 小时。训练损失从 13.47 稳定下降至 0.22。

### 5.2 Generator 模型训练

Generator 模型在 145K 条带反思 Token 的训练数据上训练 3 个 epoch，总耗时约 66.5 小时（约 27,306 步）。训练损失从 3.99 下降至 0.21，收敛曲线平滑。

![Generator 训练 Loss 曲线](figures/fig4_loss_curves.png)

### 5.3 多任务评测结果

| 任务 | 模式 | Our (Llama-2) | Official | Llama-2 | 论文报告 |
|------|------|:---:|:---:|:---:|:---:|
| PopQA | no_retrieval | 23.45% | 28.66% | — | 32.7% |
| PopQA | always_retrieve | **50.46%** | **52.32%** | — | 54.9% |
| ARC-C | no_retrieval | 57.25% | 62.29% | 43.34% | — |
| TriviaQA | no_retrieval | 31.50% | 29.75% | 17.05% | — |

**结果分析**：

1. **PopQA**：我们的模型在 always_retrieve 模式下达到 50.46%，与 Official 的 52.32% 仅差 1.86 个百分点。
2. **ARC-Challenge**：Our 模型（57.25%）显著超越原始 Llama-2（43.34%，+13.91pp），证明 Self-RAG 微调有效提升了推理能力。
3. **TriviaQA**：Our 模型（31.50%）甚至略优于 Official（29.75%），并大幅超越 Llama-2 基线（17.05%）。

![多任务评测结果对比](figures/fig1_main_results.png)

![Self-RAG 微调提升幅度](figures/fig5_improvement.png)

### 5.4 检索增强效果分析

| 模型 | No Retrieval | Always Retrieve |
|------|:---:|:---:|
| Our (Llama-2) | 23.45% | 50.46% (**+27.01pp**) |
| Official | 28.66% | 52.32% (**+23.66pp**) |

检索增强带来了 23–27 个百分点的准确率提升，证明了 Self-RAG 在知识密集型任务上的有效性。

![PopQA 检索增强效果对比](figures/fig2_retrieval_effect.png)

---

## 6. 消融实验

为了定量分析各反思 Token 评分机制的独立贡献，我们在 PopQA always_retrieve 模式下设计了四组消融实验：

| 变体 | 准确率 | Δ vs Full | 说明 |
|------|:---:|:---:|------|
| Full (G+U) | **50.46%** | — | Groundness + Utility 双评分 |
| w/o Groundness | 46.25% | −4.22pp | 仅使用 Utility 评分排序 |
| w/o Utility | 50.54% | +0.07pp | 仅使用 Groundness 评分排序 |
| w/o All Scoring | 46.32% | −4.15pp | 不进行评分排序 |

**关键发现**：

1. **Groundness 是核心贡献因子**。去掉 Groundness 评分后准确率从 50.46% 降至 46.25%（−4.22pp），说明 Groundness 在候选回答排序中发挥了决定性作用。

2. **Utility 对短答案 QA 几乎无影响**。去掉 Utility 后准确率为 50.54%，与 Full 基本一致（+0.07pp）。

3. **w/o All ≈ w/o Groundness**。两者几乎一致（46.32% vs 46.25%），进一步证实了 Utility 在 PopQA 上的独立贡献为零。

4. **与论文趋势一致**。原论文 Table 3 同样呈现 Groundness > Utility 的贡献顺序。

![反思 Token 消融实验结果](figures/fig3_ablation.png)

---

## 7. 改进实验

### 7.1 跨架构泛化: Qwen2.5-7B

为验证 Self-RAG 框架的跨架构泛化能力，我们将 Generator 的基座模型从 Llama-2-7B 替换为 Qwen2.5-7B。Qwen2.5 具有更大的词表（152K vs 32K）和不同的分词器架构（tiktoken BPE vs SentencePiece）。

**工程适配**：原始 `finetune.py` 仅支持 Llama/GPTNeoX/GPT2 三类 tokenizer 的 special token 注册，导致 Qwen2.5 训练时崩溃。我们添加了通用 tokenizer 的 else 分支。

**训练曲线对比**：Qwen2.5 的初始 Loss（6.57）高于 Llama-2（3.99），因为其词表更大。训练完成后最终 Loss 为 0.05，显著低于 Llama-2 的 0.21。训练过程中因 NAS 磁盘满导致中断（Step 21,635，79%），通过 `--resume_from_checkpoint epoch_1` 成功恢复并完成全部 3 个 epoch（总耗时约 69 小时）。

**跨架构评测结果**：

| 任务 | 模式 | Qwen2.5 | Our (Llama-2) | Official |
|------|------|:---:|:---:|:---:|
| PopQA | no_retrieval | 19.44% | 23.45% | 28.66% |
| PopQA | always_retrieve | 49.68% | 50.46% | 52.32% |
| ARC-C | no_retrieval | **68.09%** | 57.25% | 64.25% |
| TriviaQA | no_retrieval | 26.80% | 31.50% | 38.80% |

**分析**：

1. **ARC-C：Qwen2.5 大幅领先（+10.84pp）**。Qwen2.5 在科学推理任务上达到 68.09%，不仅远超 Llama-2（57.25%），甚至超越了 Official 模型（64.25%）。这表明 Qwen2.5 更强的预训练推理能力通过 Self-RAG 微调得以保留和放大。

2. **PopQA always_retrieve：基本持平（−0.79pp）**。检索增强模式下两种基座性能差异极小，说明 Self-RAG 的增益主要来源于检索段落而非模型内部知识。

3. **知识召回类任务下降**。PopQA NR 和 TriviaQA 分别下降 4.00pp 和 4.70pp。训练数据全为英文，而 Qwen2.5 的 152K 词表为中英混合设计，英文知识密度被稀释。

**关键结论**：Self-RAG 的跨架构泛化是**任务相关的**——推理类任务受益于更强的基座，而知识召回类任务更依赖基座的英文知识密度。

![跨架构评测对比](figures/fig6_cross_arch.png)

### 7.2 交互式 Demo 系统

我们基于 Gradio + vLLM 构建了交互式 Demo 系统：
- **反思 Token 可视化**：对模型输出中的反思 Token 进行颜色编码
- **vLLM 推理加速**：使用 vLLM 引擎进行高效推理，GPU 显存占用约 37GB
- **SSH 隧道访问**：通过 `ssh -L 7860:localhost:7860` 映射服务器端口

下图展示了系统的初始界面。左侧为用户输入区与快捷测试用例，右侧为反思 Token 图例及模型响应区。图例使用不同颜色对四类反思 Token 进行了编码：蓝色表示检索决策（`[Retrieval]` / `[No Retrieval]`），绿色表示质量评估（`[Relevant]` / `[Fully supported]`），紫色表示效用评分（`[Utility:1~5]`）。

**图7：Self-RAG 交互式 Demo 系统初始界面**

![Demo 初始界面](figures/1.png)

下方展示了两个实际交互案例。左图中，用户询问量子计算的概念，模型首先生成 `[Retrieval]` 信号表明需要外部知识辅助，随后给出解释并标注 `[Fully supported]`（完全支持），最终自评 `[Utility:5]`（最高分）。右图中，用户要求编写 Fibonacci 函数，模型同样触发了 `[Retrieval]`，但对自身生成的代码仅给出 `[Partially supported]`（部分支持），体现了 Self-RAG 诚实的自我批判能力。

**图8：Self-RAG Demo 交互实例——反思 Token 可视化效果**

| 量子计算解释：触发检索 + 完全支持 | 代码生成：触发检索 + 部分支持 |
|:---:|:---:|
| ![量子计算](figures/2.png) | ![Fibonacci 代码](figures/3.png) |

---

## 8. 工程挑战与解决方案

在复现和改进实验过程中，我们遇到并解决了 21 项工程问题。以下为代表性问题：

| 编号 | 问题 | 根因 | 解决方案 |
|:---:|------|------|---------|
| P5 | AdamW 导致 OOM | 优化器状态 ~24GB | 替换为 Adafactor（节省 ~12GB） |
| P16 | vLLM logprobs 超限 | vLLM 0.5.5 限制 | 在评测脚本中移除 logprobs 参数 |
| P19 | PopQA 虚假 100% | match 指标 + 长文本 | 改用 always_retrieve 模式 |
| P20 | ctxs 字段缺失 | 数据不含预检索段落 | 从 HuggingFace 下载完整数据集 |
| P21 | Qwen tokenizer 不兼容 | finetune.py 仅处理 Llama | 添加通用 tokenizer else 分支 |

完整的 21 项问题记录及详细解决过程见项目文档 `problem.md`。

---

## 9. 结论

本报告对 Self-RAG 进行了系统性的复现与改进实验，主要结论如下：

1. **复现验证**：在单卡 A40 上成功复现了 Self-RAG 的完整训练流程（Critic 21.5h + Generator 66.5h），在三个基准上的评测结果与原论文保持一致。

2. **检索增强效果显著**：在 PopQA 上，always_retrieve 模式相比 no_retrieval 提升了 27 个百分点（23.45% → 50.46%）。

3. **Groundness 是核心机制**：消融实验表明，Groundness 评分贡献了 4.22 个百分点的准确率提升，是检索增强有效性的关键。

4. **跨架构泛化**：成功将 Self-RAG 迁移至 Qwen2.5-7B。Qwen2.5 在推理任务 ARC-C 上达到 68.09%（+10.84pp），但在知识召回任务上有所下降，揭示了 Self-RAG 泛化的任务依赖性。

5. **工程实践**：系统记录了 21 项工程问题，涵盖显存优化、数据处理、跨架构兼容等方面。

---

## 10. 局限性与未来工作

尽管 Self-RAG 在知识密集型任务上表现出显著优势，本方法和本项目复现实验仍存在若干局限。

1. **训练与推理成本较高。** Self-RAG 不仅需要训练 Generator，还需要先训练或使用 Critic 对数据进行反思 Token 标注。对于 7B 级别模型，即使在单卡 A40 上使用 BF16、Adafactor 和 Gradient Checkpointing，完整训练仍需要数十小时。推理阶段如果启用 always_retrieve，还需要对多个检索段落分别生成候选回答并打分，吞吐成本明显高于普通生成。

2. **依赖高质量检索上下文。** Self-RAG 的优势很大程度上建立在检索段落质量之上。如果评测数据缺少 `ctxs` 字段，或检索器返回的段落与问题弱相关，always_retrieve 模式会退化甚至引入额外噪声。本项目在 PopQA 上曾遇到预检索段落缺失导致评测结果异常的问题，说明数据完整性和检索质量是复现实验中的关键前提。

3. **反思 Token 的监督信号仍有噪声。** 原论文使用 GPT-4 产生 Critic 标注，本项目由于官方 Google Drive 数据失效，部分数据获取和处理需要依赖 HuggingFace 替代源与工程化修复。自动标注虽然降低了人工成本，但反思 Token 的质量仍受标注模型、提示模板和数据分布影响。

4. **评测指标存在任务适配问题。** 本实验统一使用 match accuracy，即生成文本与参考答案子串匹配。该指标适合短答案 QA，但对长文本输出可能产生误判，尤其在 PopQA 中曾出现长文本与短别名答案偶然匹配导致虚假高分的问题。因此，Self-RAG 在摘要、传记生成等长文本任务上的评价仍需要更细粒度的事实一致性指标。

5. **跨架构迁移具有任务依赖性。** Qwen2.5-7B 在 ARC-C 上显著优于 Llama-2-7B，但在 PopQA no_retrieval 和 TriviaQA 上低于 Llama-2。这说明更强的通用推理基座不必然带来更好的英文知识召回性能，Self-RAG 的收益会受到基座模型预训练语料、词表设计和任务类型共同影响。

基于上述局限，后续可以从四个方向拓展：第一，在更多基座模型（如 Mistral-7B、Llama-3、Qwen3 等）上系统验证 Self-RAG 的跨架构规律；第二，在中文 QA 和中英混合检索场景中构造反思 Token 数据，评估该框架对中文 NLP 任务的适配能力；第三，将 Contriever 或其他在线检索器接入推理链路，从预检索评测扩展到真正端到端的自适应 RAG 系统；第四，在长文本生成任务上重新评估 Utility 评分的价值，并引入事实一致性、引用准确率和人工评价等更全面的指标。

---

## 11. 组员分工

本项目由叶盛豪、闪瑜两位组员合作完成，整体工作量基本均衡，各占 50%。具体分工如下：

| 组员 | 学号 | 主要负责内容 |
|------|------|-------------|
| 叶盛豪 | PB24000227 | 负责 Self-RAG 原论文与开源代码阅读、实验环境搭建、Llama-2-7B Critic 与 Generator 训练复现、主实验评测结果整理、报告 LaTeX 框架撰写与图表集成。 |
| 闪瑜 | PB23051016 | 负责训练与评测数据下载及预处理、PopQA/ARC-C/TriviaQA 评测脚本调试、反思 Token 消融实验设计与执行、Qwen2.5-7B 跨架构迁移实验、交互式 Demo 与工程问题文档整理。 |
| 共同完成 | — | 共同完成论文方法讨论、实验方案设计、结果分析、报告文字修改与最终材料整理。 |

项目开源仓库地址为：https://github.com/SunTomb/NLP。

---

## 参考文献

1. Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2024). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. *ICLR 2024*.
2. Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020*.
3. Guu, K., et al. (2020). REALM: Retrieval-Augmented Language Model Pre-Training. *ICML 2020*.
4. Izacard, G., et al. (2023). Atlas: Few-shot Learning with Retrieval Augmented Language Models. *JMLR*.
5. Jiang, Z., et al. (2023). Active Retrieval Augmented Generation. *EMNLP 2023*.
6. Bai, Y., et al. (2022). Constitutional AI: Harmlessness from AI Feedback. *arXiv:2212.08073*.
7. Madaan, A., et al. (2023). Self-Refine: Iterative Refinement with Self-Feedback. *NeurIPS 2023*.
8. Touvron, H., et al. (2023). Llama 2: Open Foundation and Fine-Tuned Chat Models. *arXiv:2307.09288*.
9. Qwen Team. (2024). Qwen2.5: A Party of Foundation Models. *arXiv:2412.15115*.
