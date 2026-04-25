# Self-RAG 项目问题记录与解决方案

> 本文档记录了 Self-RAG 复现项目在实验室集群 (Tang-2-Wu, 8×A40) 上部署与运行过程中遇到的所有有价值的技术问题及其解决方案。
> 适用于实验报告的"问题与挑战"章节，也可作为后续类似项目的工程参考。

---

## 目录

1. [环境与依赖](#1-环境与依赖)
2. [模型管理与存储](#2-模型管理与存储)
3. [推理与验证](#3-推理与验证)
4. [数据获取](#4-数据获取)
5. [训练调试](#5-训练调试)
6. [评测调试](#6-评测调试)

---

## 1. 环境与依赖

### P1: Conda 环境隔离 —— `--prefix` vs `--name`

**问题描述**：
实验室集群的 home 目录配额有限（通常 < 20GB），而 Self-RAG 的 conda 环境（PyTorch + vLLM + Transformers）安装后超过 15GB。使用默认的 `conda create --name` 会将环境安装到 `~/.conda/envs/`，很快耗尽 home 配额。

**解决方案**：
使用 `--prefix` 指定环境安装到 NAS 工作目录下：

```bash
conda create --prefix /NAS/yesh/NLP/.conda/selfrag python=3.10 -y
conda activate /NAS/yesh/NLP/.conda/selfrag
```

**额外措施**：将 HuggingFace 缓存也重定向到 NAS：

```bash
export HF_HOME=/NAS/yesh/hf_cache
export TRANSFORMERS_CACHE=/NAS/yesh/hf_cache/hub
```

**工程经验**：
- `--prefix` 安装的环境激活时需要完整路径，可封装到 `activate.sh` 脚本中
- 建议在项目根目录创建统一的环境激活脚本，包含所有 PATH 和环境变量设置

---

### P2: vLLM 导入失败 —— `pyairports` 空壳包问题

**问题描述**：
安装 vLLM 0.5.5 后，`from vllm import LLM` 抛出 `ImportError`：

```
File ".../outlines/types/airports.py", line 7, in <module>
    raise ImportError(
ImportError: The `airports` module requires "pyairports" to be installed.
```

pip 显示 `pyairports` 已安装（0.0.1），但 `import pyairports` 仍然失败。

**根因分析**：

经过逐步排查，发现 `pyairports` 0.0.1 是一个 **PyPI 占位空壳包**：

| 特征 | 值 |
|------|-----|
| 作者 | `John Doe` |
| Email | `males-folds0a@icloud.com` |
| Summary | 空 |
| 实际 Python 文件 | **零** — 只有 `dist-info` 元数据 |

```bash
# 验证：只有元数据，没有任何 .py 文件
$ find .../site-packages/pyairports* -type f
.../pyairports-0.0.1.dist-info/METADATA
.../pyairports-0.0.1.dist-info/WHEEL
.../pyairports-0.0.1.dist-info/RECORD
# 没有 __init__.py，没有 airports.py
```

这是一个 **PyPI 命名空间占位（name squatting）** 的典型案例：有人注册了 `pyairports` 包名但没有发布实际代码。`outlines` 库（vLLM 的依赖）期望从 `pyairports.airports` 导入 `AIRPORT_LIST`，但该模块根本不存在。

**排查过程**：

1. ❌ 首先尝试 `pip install pyairports` — pip 报告已安装，但导入失败
2. ❌ 尝试从官方 PyPI 安装更新版本：`pip install 'pyairports>=2.0' --index-url https://pypi.org/simple/` — PyPI 上只有 0.0.1
3. ❌ 尝试降级 `outlines`：`pip install 'outlines==0.0.44'` — 该版本同样依赖 `pyairports`
4. ❌ 强制重装：`pip install --force-reinstall --no-cache-dir pyairports` — 重装后仍然是空壳
5. ✅ **最终方案：手动创建模块文件**

**解决方案**：

```bash
# 创建 pyairports 包目录
mkdir -p /NAS/yesh/NLP/.conda/selfrag/lib/python3.10/site-packages/pyairports

# 创建 __init__.py
echo "" > .../pyairports/__init__.py

# 创建 airports.py，提供 outlines 需要的 AIRPORT_LIST
echo "AIRPORT_LIST = []" > .../pyairports/airports.py

# 验证
python -c "from vllm import LLM; print('vllm import OK ✅')"
```

**依赖链路图**：

```
vllm 0.5.5
  └── outlines >=0.0.43,<0.1
        └── outlines/types/airports.py
              └── from pyairports.airports import AIRPORT_LIST  ← 💥 空壳包
```

**启示**：
- PyPI 生态存在命名空间占位问题，`pip install` 成功不代表包可用
- 在受限网络环境（如校园/实验室集群）中，镜像站可能缓存了有问题的包版本
- 对于非核心依赖（如 vLLM 的 airports 类型检查），手动创建 stub 模块是合理的工程解决方案
- 排查时应检查包的 **实际文件内容**（`find ... -type f`），而不仅仅依赖 `pip show`

---

### P3: USTC 镜像站版本滞后

**问题描述**：
集群配置了 USTC PyPI 镜像（`https://mirrors.ustc.edu.cn/pypi/web/simple`）。在排查 P2 时，尝试指定官方 PyPI 源安装更新版本：

```bash
pip install 'pyairports>=2.0' --index-url https://pypi.org/simple/
# ERROR: No matching distribution found for pyairports>=2.0
```

结果发现官方 PyPI 同样只有 0.0.1（因为这就是一个空壳包，不存在更新版本）。

**经验总结**：
- 遇到依赖安装问题时，先检查官方 PyPI 是否有目标版本
- 镜像站问题和包本身问题要分开排查
- 可以使用 `pip install --index-url https://pypi.org/simple/` 绕过镜像站验证
- 集群上建议在 `pip.conf` 中同时配置镜像站和官方源作为 fallback

---

## 2. 模型管理与存储

### P4: Meta Llama 2 Gated Repo 访问被拒

**问题描述**：
尝试从官方仓库 `meta-llama/Llama-2-7b-hf` 下载模型时，请求被拒绝：

```
Your request to access this repo has been rejected by the repo's authors.
```

原因是 Meta 的 Llama 2 模型属于 **Gated Repository**，需要在 HuggingFace 上填写访问申请表。由于申请表填写不规范，请求被拒。

**解决方案**：
使用社区发布的等价镜像仓库 `NousResearch/Llama-2-7b-hf`：

```bash
huggingface-cli download NousResearch/Llama-2-7b-hf \
    --local-dir models/Llama-2-7b-hf
```

**对比**：

| 来源 | 优劣 |
|------|------|
| `meta-llama/Llama-2-7b-hf` | 官方权重，需要审批 |
| `NousResearch/Llama-2-7b-hf` | 社区镜像，权重完全一致，无需审批 |

**经验总结**：
- Gated Repo 的申请需要认真填写组织信息和使用目的
- 大部分开源模型都有社区镜像，可作为备选下载源
- 在实验报告中应注明实际使用的模型来源

---

### P5: 大模型存储策略 —— Symlink 避免重复存储

**问题描述**：
Self-RAG 需要至少两个 7B 模型（Self-RAG + Llama 2 基座），每个约 14GB，共 ≥28GB。如果在 HF 缓存和项目目录中各存一份，将浪费约 28GB 空间。NAS 虽然空间较大但也不应滥用。

**解决方案**：
采用 **单一存储 + 符号链接** 策略：

```
/NAS/yesh/hf_cache/hub/          ← 唯一存储（HuggingFace 缓存格式）
    models--selfrag--selfrag_llama2_7b/
    models--NousResearch--Llama-2-7b-hf/

/NAS/yesh/NLP/models/            ← 符号链接（项目引用）
    selfrag_llama2_7b -> /NAS/yesh/hf_cache/hub/.../snapshots/xxx
    Llama-2-7b-hf    -> /NAS/yesh/hf_cache/hub/.../snapshots/xxx
```

创建链接的命令：

```bash
SNAP=$(find /NAS/yesh/hf_cache/hub/models--selfrag--selfrag_llama2_7b/snapshots/ \
    -mindepth 1 -maxdepth 1 -type d | head -1)
ln -s "$SNAP" models/selfrag_llama2_7b
```

**优点**：
- 节省 ~28GB 存储空间
- 模型更新时只需更新缓存
- 多项目可共享同一份权重

**注意事项**：
- 符号链接内部的 blob 文件本身也是符号链接（HF Cache 结构），需确保整个链路完整
- 训练产生的新 checkpoint 应单独保存到 `outputs/` 目录，不要污染原始模型目录

---

### P6: HuggingFace 下载并发锁冲突

**问题描述**：
使用 `huggingface-cli download` 下载模型时，出现大量锁等待信息：

```
Still waiting to acquire lock on .cache/huggingface/.gitignore.lock (elapsed: 0.2s)
```

多个分片的下载进程同时尝试创建/修改缓存目录中的 `.gitignore` 文件，导致文件锁竞争。

**解决方案**：
这是正常现象，等待即可。如果持续卡死：
- 手动删除锁文件：`rm -f models/.../.cache/huggingface/*.lock`
- 或使用 `--local-dir-use-symlinks False`（新版已弃用此参数）

---

## 3. 推理与验证

### P7: 推理验证 —— 反思 Token 行为符合预期

**验证过程**：
使用 7 个精心设计的测试用例，涵盖不同场景，验证 Self-RAG 模型的反思 Token 生成行为。

**验证结果**：

| 测试场景 | Query | 预期反思 Token | 实际输出 | 结果 |
|----------|-------|---------------|---------|:----:|
| 简单事实（2+2） | What is 2+2? | 不检索 | `4[Utility:5]` | ✅ |
| 常识推理 | Leave odd one out | `[No Retrieval]` | `[No Retrieval]`...推理过程 | ✅ |
| 知识问题（无段落） | llama vs alpaca | `[Retrieval]` | `[Retrieval]`...自行编造段落 | ✅ |
| 知识问题（有段落） | llama vs alpaca + paragraph | `[Relevant]+[Fully supported]` | 完全匹配 | ✅ |
| 科学问题 | overfitting + evidence | `[Relevant]+[Fully supported]` | 完全匹配 | ✅ |
| 中文测试 | 中国的首都 | 有限支持 | `首都是北京。[Utility:5]` | ✅✓ |
| 事实核查 | 8 glasses of water | `[Retrieval]` | 先检索后自主判断 | ✅ |

**关键发现**：
- 模型对简单问题自动跳过检索（`[No Retrieval]`），符合论文描述的 adaptive retrieval 机制
- 提供段落时，模型能正确评估相关性和支持度
- `[Utility:5]` 全部为满分，说明模型对自身回答质量有较高自信
- Llama 2 对中文有基本支持能力，但未触发检索机制

**意义**：
这一验证确认了预训练 Self-RAG 模型在我们的集群环境中能正确工作，为后续的 Critic/Generator 训练建立了 baseline。

---

## 4. 数据获取

### P8: 官方 Google Drive 数据链接全部失效

**问题描述**：
Self-RAG 官方仓库中的三个关键 Google Drive 下载链接均返回"文件未找到"：

| 数据 | Google Drive File ID | 状态 |
|------|---------------------|------|
| Critic 训练数据 | `1IN1XcIOYtRIGWITJ4LKRgfITT-uUwk_W` | ❌ 失效 |
| Generator 训练数据 (150K) | `10G_FozUV4u27EX0NjwVe-3YMUMeTwuLk` | ❌ 失效 |
| 评测数据集 | `1TLKhWjez63H4uBtgCxyoyJsZi-IMgnDb` | ❌ 失效 |

**影响范围**：

```
├── Critic 训练 ← 无法获取 GPT-4 标注的反思 Token 训练数据
├── Generator 训练 ← 已有 HuggingFace 替代源 ✅
└── 评测 ← 无法获取包含预检索段落的评测数据
```

**解决方案**：

**1. Generator 训练数据**（已解决）：
作者同时上传到了 HuggingFace Datasets：

```bash
huggingface-cli download selfrag/selfrag_train_data \
    --repo-type dataset --local-dir data/generator/
```

成功下载 145,619 条训练数据（256 MB）。

**2. Critic 训练数据**（合成解决）：
由于原始数据由 GPT-4 标注，无法直接获取替代源。我们编写了 `scripts/prepare_data.py` 脚本，从 Generator 训练数据中**逆向提取**反思 Token，合成等价的 Critic 训练数据：

- 核心思路：Generator 训练数据中已包含 `[Retrieval]`、`[Relevant]`、`[Fully supported]`、`[Utility:5]` 等反思 Token。我们将其提取出来，按 Critic 模型的训练格式（`instruction + input → output` 分类任务）重新组织
- 覆盖 4 种任务：retrieval 决策、relevance 判断、groundedness 评估、utility 评分
- 格式与官方 `combine_chat_gpt_reward.py` 脚本输出一致

```bash
python scripts/prepare_data.py --task critic
# 输出: data/critic/critic_train_data_train.json + _dev.json
```

**3. 评测数据集**（从 HuggingFace 下载原始数据集并处理）：
使用 `datasets` 库从 HuggingFace 下载原始公开数据集：

```bash
python scripts/prepare_data.py --task eval
# 下载: PopQA, ARC-Challenge, PubHealth, TriviaQA
```

**注意**：官方评测数据包含预检索段落（使用 Contriever-MSMARCO 检索），我们下载的原始数据集不包含。后续评测需要先运行检索器添加上下文，或使用 `no_retrieval` 模式进行对比实验。

**经验总结**：
- 学术论文的数据链接（尤其是 Google Drive）有过期风险，使用时应第一时间备份
- HuggingFace Datasets 比 Google Drive 更可靠，建议优先使用
- 对于 GPT-4 标注数据，可以通过已有的下游数据逆向合成，虽然质量略有差异但对于复现实验足够
- 在实验报告中应如实说明数据来源差异及其可能对结果的影响

---

## 5. 训练调试

### P9: 相对导入失败 —— `train_special_tokens.py` 无法直接运行

**问题描述**：
执行 Critic 训练脚本时报错：

```
File "self-rag/data_creation/train_special_tokens.py", line 27
    from ..retrieval_lm.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
ImportError: attempted relative import with no known parent package
```

**根因分析**：
`train_special_tokens.py` 使用了 `from ..retrieval_lm` 的**相对导包**语法，这要求脚本作为 package 的一部分通过 `python -m` 方式调用。直接用 `python train_special_tokens.py` 运行时，Python 无法确定父包位置。

**解决方案**：
由于该导入仅用于 `flash_attn` 猴子补丁（替换 Llama 的注意力实现），而我们不使用 flash attention，直接注释掉即可：

```python
# 第 27 行：注释掉
#from ..retrieval_lm.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

# 第 287 行：注释掉调用，加 pass 避免空 if 块
if model_args.use_flash_attn:
    #replace_llama_attn_with_flash_attn()
    pass
```

**经验总结**：
- 学术项目的代码通常假设特定的运行方式（如从项目根目录执行），需要注意调用路径
- 对于非核心功能（如 flash_attn），注释掉比修复导入路径更安全
- `use_flash_attn` 默认为 `False`，注释后不影响训练正确性

---

### P10: PROMPT_DICT 键缺失 —— 训练数据格式不匹配

**问题描述**：
修复导入问题后，训练报 `KeyError: 'prompt_no_input_paragraph'`：

```python
# 第 236 行尝试解包 4 个键，但 PROMPT_DICT 只定义了 2 个
prompt_input, prompt_no_input, prompt_no_input_paragraph, prompt_no_input_separated = \
    PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"], \
    PROMPT_DICT["prompt_no_input_paragraph"], PROMPT_DICT["prompt_no_input_separated"]
```

**解决方案**：
在 `PROMPT_DICT` 中补充缺失的模板（使用与 `prompt_no_input` 相同的格式）：

```python
PROMPT_DICT = {
    "prompt_input": "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:",
    "prompt_no_input": "### Instruction:\n{instruction}\n\n### Response:",
    "prompt_no_input_paragraph": "### Instruction:\n{instruction}\n\n### Response:",
    "prompt_no_input_separated": "### Instruction:\n{instruction}\n\n### Response:",
}
```

---

### P11: FSDP 与 gradient_checkpointing 冲突 —— 冗余 AllGather

**问题描述**：
使用 4 卡 FSDP 训练时，即使 GPU 有充足剩余显存仍 OOM。日志中出现明确警告：

```
When using FSDP full shard, instead of using `gradient_checkpointing` in TrainingArguments,
please use `activation_checkpointing` in `fsdp_config`. The former introduces a redundant
AllGather operation in backward pass.
```

**根因分析**：
在 FSDP 模式下，`--gradient_checkpointing True`（HF TrainingArguments 参数）会在反向传播时触发**冗余的 AllGather 操作**。AllGather 操作会将分片的参数重新聚合到每张 GPU 上，导致每张卡在反向传播时临时需要完整模型的参数副本，抵消了 FSDP 分片带来的显存节省。

**解决方案**：
如果使用 FSDP，应通过 `fsdp_config` 配置 activation checkpointing，而非使用 TrainingArguments 的 `gradient_checkpointing`：

```bash
# ❌ 错误用法（与 FSDP 冲突）
--fsdp "full_shard auto_wrap" --gradient_checkpointing True

# ✅ 正确用法
--fsdp "full_shard auto_wrap" --fsdp_config '{"activation_checkpointing": true}'

# ✅ 或者：单卡训练时可直接使用 gradient_checkpointing
# （不使用 FSDP，无冲突）
--gradient_checkpointing True
```

**经验总结**：
- FSDP 和 HF Trainer 的 checkpoint 机制有两套独立实现，混用会产生冲突
- 遇到多卡训练 OOM 时，不仅要看绝对显存数值，还要关注框架间的兼容性警告

---

### P12: 模型默认以 fp32 加载 —— 所有 OOM 的根因（★ 最关键）

**问题描述**：
在单卡 A40（46GB 完全空闲）上用 `--bf16 True --optim adafactor --gradient_checkpointing True` 训练，理论上只需 ~30GB 显存，但实际占用 **44.3 GB** 后 OOM。

**根因分析**：
这是**所有 Critic 训练 OOM 问题的根本原因**。

`train_special_tokens.py` 第 296 行的模型加载代码：

```python
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_args.model_name_or_path,
    cache_dir=training_args.cache_dir,
)  # ← 没有指定 torch_dtype！
```

`--bf16 True` 只控制训练时的 **autocast**（前向计算用 bf16），但 `from_pretrained` 默认以 **fp32** 加载模型权重。这导致：

| 组件 | fp32（实际加载） | bf16（预期） |
|------|:--------------:|:----------:|
| 模型参数 | **26.8 GB** | 13.4 GB |
| 梯度 | **26.8 GB** | 13.4 GB |
| 合计 | **53.6 GB** | 26.8 GB |

即使使用 Adafactor（优化器仅 ~0.1GB）+ gradient_checkpointing，53.6 GB 的参数 + 梯度本身就超过了 A40 的 46 GB。

**解决方案**：
在 `from_pretrained` 中显式指定 `torch_dtype`：

```python
# 根据训练精度设置模型加载 dtype
if training_args.bf16:
    model_dtype = torch.bfloat16
elif training_args.fp16:
    model_dtype = torch.float16
else:
    model_dtype = None
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_args.model_name_or_path,
    cache_dir=training_args.cache_dir,
    torch_dtype=model_dtype,  # ← 关键修复
)
```

**修复后效果**：

| 指标 | 修复前 | 修复后 |
|------|:------:|:------:|
| 模型显存 | 26.8 GB (fp32) | 13.4 GB (bf16) |
| 总显存 | 44.3 GB → OOM | ~22 GB |
| 训练状态 | ❌ 崩溃 | ✅ 成功 |
| Loss 下降 | — | 8.43 → 5.16 (↓39%) |

**经验总结**：
- HuggingFace `from_pretrained` 的默认行为是 **fp32 加载**，即使 checkpoint 文件存储的是 fp16/bf16
- `--bf16 True` 只影响训练过程的 autocast，**不影响模型加载精度**
- 在 7B 级别模型上，fp32 vs bf16 的显存差距是 **26.8 GB**，足以决定训练成败
- 这类问题在小模型（如 BERT、GPT-2）上不明显，但在 LLM 时代是必须注意的关键细节
- **教训**：OOM 排查时，第一步应检查模型实际加载的 dtype，而不是假设它与 `--bf16` 参数一致

---

### P13: NCCL 超时 —— tokenize 阶段各 rank 速度差异

**问题描述**：
使用 4 卡 FSDP 训练 Generator 时，在 tokenize 145K 条数据阶段 NCCL 超时崩溃：

```
torch.distributed.DistBackendError: NCCL communicator was aborted on rank 2.
Original reason for aborting was: watchdog callback timed out
```

**根因分析**：
`finetune.py` 使用 `accelerator.main_process_first()` + `datasets.map()` 进行 tokenize。虽然 `main_process_first()` 意图让主进程先处理、其他进程读缓存，但实际上每个 rank 可能都独立 tokenize 了 145K 条（缓存目录冲突）。由于 4 个进程在不同 GPU 上速度有差异，最快的 rank 完成 tokenize 后等待其他 rank，**超过默认 600 秒（10 分钟）超时**后 NCCL 通信中断。

**解决方案**：
在 `finetune.py` 中增大 NCCL 超时时间到 30 分钟：

```python
# 1. 环境变量层面
os.environ.setdefault("NCCL_TIMEOUT", "1800")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "0")

# 2. Accelerator 初始化层面（关键）
from accelerate.utils import InitProcessGroupKwargs
import datetime
process_group_kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=1800))
accelerator = Accelerator(
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    kwargs_handlers=[process_group_kwargs],
    **accelerator_log_kwargs
)
```

**经验总结**：
- 仅设置环境变量 `NCCL_TIMEOUT` 不够，必须通过 `InitProcessGroupKwargs` 在代码层面传递
- 大规模数据 tokenize（>100K 条）是 FSDP 训练的隐形瓶颈
- 更优方案是预先 tokenize 数据并保存为 Arrow 格式（`dataset.save_to_disk()`），完全避免训练时 tokenize

---

### P14: FSDP 下 AdamW OOM —— 优化器状态超出显存

**问题描述**：
解决 P13 后 tokenize 成功完成，但第一个 `optimizer.step()` 时 OOM：

```
[rank2]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 152.00 MiB.
GPU 2 has a total capacity of 44.35 GiB of which 140.12 MiB is free.
```

**根因分析**：
AdamW 需要为每个参数维护两个 fp32 状态（`exp_avg` + `exp_avg_sq`）。在 FSDP FULL_SHARD 模式下：

| 组件 | 每卡显存 |
|------|---------|
| 主权重 fp32（FSDP 自动上转） | ~6.7 GB |
| 梯度 fp32 | ~6.7 GB |
| AdamW exp_avg (fp32) | ~6.7 GB |
| AdamW exp_avg_sq (fp32) | ~6.7 GB |
| 前向 bf16 参数（allgather 重建） | ~13.4 GB |
| 激活值（gradient_checkpointing） | ~8-15 GB |
| **总计** | **~49-56 GB > 44 GB** |

FSDP mixed precision 的 `Upcasted low precision parameters` 警告正是关键线索——FSDP 会将 bf16 参数上转为 fp32 存储主权重。

**解决方案**：
将 AdamW 替换为 Adafactor：

```python
from transformers import Adafactor
optimizer = Adafactor(
    optimizer_grouped_parameters,
    lr=args.learning_rate,
    scale_parameter=False,
    relative_step=False,
    warmup_init=False,
)
```

Adafactor 将 `exp_avg_sq` 分解为行向量和列向量，每卡节省 ~12 GB。

---

### P15: FSDP 4 卡反而比单卡慢 4.5 倍（★ 意外发现）

**问题描述**：
解决 P13 + P14 后，4 卡 FSDP + Adafactor 成功启动训练。但观察到：

| 方案 | 速度 | 每卡显存 | 预计总耗时 |
|------|------|---------|-----------|
| 4 卡 FSDP + Adafactor | **~40 s/step** | ~40+ GB (接近上限) | **303 小时 ❌** |
| 单卡 Adafactor | **~8.8 s/step** | ~35 GB | **66 小时 ✅** |

4 卡 FSDP 比单卡**慢了 4.5 倍**！

**根因分析**：

1. **FSDP allgather/reduce_scatter 通信开销**：FULL_SHARD 模式下，每次前向和反向都需要通过 NVLink/PCIe 在 4 卡间传输完整参数。对于 7B 模型，每次 allgather 需传输 ~13 GB 数据
2. **fp32 upcast 开销**：FSDP mixed precision 将所有 bf16 参数上转为 fp32 存储，增加了内存带宽和计算负担
3. **显存接近上限**：每卡 40+ GB / 44 GB，PyTorch 频繁执行 CUDA malloc retry，导致 GPU 实际利用率低
4. **4 进程竞争共享资源**：NVLink 带宽、PCIe 带宽、CPU 内存等均被 4 进程共享

**结论**：
对于 7B 模型 + 44GB GPU 的配置，FSDP 的通信开销和内存压力远大于数据并行带来的吞吐提升。**单卡 Adafactor 是最优解**。

**适用条件表**：

| 场景 | 推荐方案 |
|------|---------|
| 7B 模型 + 44GB GPU | **单卡 Adafactor** |
| 7B 模型 + 80GB GPU | 单卡 AdamW 或 2 卡 DDP |
| 13B+ 模型 + 44GB GPU | 4 卡 FSDP（单卡放不下） |
| 70B 模型 | 8 卡 FSDP + CPU offload |

**经验总结**：
- 多卡 ≠ 更快。FSDP 的收益在模型大到单卡放不下时才显现
- 显存接近上限时，PyTorch 的 malloc retry 机制会严重拖慢速度
- 在决定分布式策略前，应先在**单卡**上验证可行性
- **教训**：工程中应该 "先跑通，再优化"，不要过早引入分布式复杂度

---

## 6. 评测调试

### P16: vLLM 0.5.5 logprobs 上限 —— `ValueError: Cannot request more than 20 logprobs`

**问题描述**：
原始 `run_short_form.py` 在推理时请求 `logprobs=32016`（覆盖完整 vocab），但 vLLM 0.5.5 引入了硬限制：最多只能请求 20 个 logprobs。运行时抛出：

```
ValueError: Cannot request more than 20 logprobs per token...
```

**根因分析**：
Self-RAG 原始代码设计于 vLLM 0.4.x 时代，当时无此限制。评分逻辑只需要特定 special token（`[Relevant]`、`[Fully supported]` 等）的 logprob，总共不超过 10 个 token。

**解决方案**：
1. 将 `logprobs` 参数从 32016 降为 20（足够覆盖所有 special tokens）
2. 在 `no_retrieval` 模式下设 `logprobs=None`（该模式不需要 logprobs 评分）

```python
# no_retrieval 模式不需要 logprobs
_logprobs = None if mode == "no_retrieval" else 20
sampling_params = SamplingParams(
    temperature=0.0, top_p=1.0, max_tokens=max_new_tokens, logprobs=_logprobs)
```

**工程经验**：
- 升级推理引擎时需检查 API 兼容性，尤其是参数范围限制
- 不要盲目请求最大范围的 logprobs，按需请求即可

---

### P17: vLLM 内存自动探测 OOM —— 共享 GPU 环境下 `gpu_memory_utilization` 冲突

**问题描述**：
在有其他进程占用 GPU 的环境下，vLLM 默认 `gpu_memory_utilization=0.9` 尝试分配 90% 显存，与已占用显存冲突导致 OOM：

```
torch.cuda.OutOfMemoryError: CUDA out of memory.
```

**根因分析**：
vLLM 在初始化时会自动探测 GPU 总显存并预分配 KV cache。它**不会自动检测其他进程的显存占用**。如果 GPU 已被占用 10GB，而 vLLM 尝试分配 48GB × 0.9 = 43.2GB，则超出可用空间。

**解决方案**：
根据 GPU 实际空闲显存调整 `gpu_memory_utilization`：

```python
model = LLM(model=gpt, dtype=args.dtype,
            tensor_parallel_size=args.world_size,
            gpu_memory_utilization=0.85)  # 共享环境下适当降低
```

| 场景 | 推荐 `gpu_memory_utilization` |
|------|:---:|
| 独占 GPU（无其他进程） | 0.85-0.90 |
| GPU 有少量占用 (< 5GB) | 0.80 |
| GPU 占用较多 (> 10GB) | 0.60 |

**工程经验**：
- vLLM 的 KV cache 是**预分配**的，nvidia-smi 显示的占用包含了未实际使用的预留空间
- 在共享集群上运行推理任务前，务必先用 `nvidia-smi` 确认空闲显存

---

### P18: 评测逐条推理瓶颈 —— vLLM batch_size=1 的性能浪费

**问题描述**：
原始 `run_short_form.py` 在主循环中逐条调用 `model.generate([单个prompt])`，PopQA 14K 条数据预计耗时 ~10 小时，GPU 利用率极低。

**根因分析**：
vLLM 的核心优势在于**连续批处理（continuous batching）**：将多个请求合并调度以最大化 GPU 吞吐。逐条调用 `model.generate([1个prompt])` 等价于 batch_size=1，完全浪费了这一能力。每次调用都有 Python → CUDA 的启动开销 (~100ms)，而 GPU 计算仅需 ~10ms。

**解决方案**：
编写新的批量评测脚本 `run_eval_batch.py`，将所有 prompt 收集后一次性传入 `model.generate(all_prompts)`：

```python
# 原始：逐条推理 (~10 小时)
for row in data:
    output = model.generate([single_prompt])  # batch=1

# 优化：批量推理 (~5 分钟)
all_prompts = [build_prompt(row) for row in data]
outputs = model.generate(all_prompts)  # vLLM 内部自动调度
```

| 方案 | PopQA 14K 耗时 | 加速比 |
|------|:---:|:---:|
| 逐条推理 (batch=1) | ~10 小时 | 1× |
| 批量推理 (vLLM batch) | ~5 分钟 | **~120×** |

**工程经验**：
- vLLM 的 `generate()` 接受 prompt 列表，内部自动管理调度和内存
- 对于离线评测（非交互式），始终应该一次性传入所有 prompt
- 吞吐瓶颈不在 GPU 计算，而在 Python 调用开销和调度低效

---

### P19: PopQA 评测指标虚假 100% —— `match()` 子串匹配与长文本生成的不兼容

**问题描述**：
使用 `no_retrieval` 模式评测 PopQA 时，**所有三个模型（Our, Official, Llama2）均显示 ~100% match 准确率**，明显不合理（论文报告 Self-RAG ~54.9%）。

**根因分析**：
`metrics.py` 中的 `match()` 函数使用**子串匹配**：

```python
def match(prediction, ground_truth):
    for gt in ground_truth:
        if gt in prediction:  # 只要 gt 是 prediction 的子串就算匹配
            return 1
    return 0
```

两个因素叠加导致虚假高分：
1. **长文本输出**：`no_retrieval` 模式下模型生成完整句子（如 "George Rankin is a retired teacher."），文本较长
2. **多个短答案**：PopQA 的 `answers` 字段包含多个同义答案（如 `["politician", "political leader", "polit.", "pol"]`），其中存在极短字符串

结果：几乎任何生成文本都包含某个短答案字符串，导致 match = 1。

**解决方案**：
使用 `always_retrieve` 模式重新评测。在该模式下：
- 模型使用检索段落辅助生成，输出更短更精确
- 通过 Relevance/Groundedness/Utility 评分选择最佳答案
- `match()` 指标在短文本上的假阳性率大幅降低

**工程经验**：
- **评测指标必须与生成模式匹配**：子串匹配适用于短答案生成，不适用于长文本生成
- 论文的评测数据大多包含 gold standard 段落（文件名 `_w_gs`），应使用检索模式
- 100% 准确率是一个明确的红旗信号，应立即排查指标合理性

---

### P20: PopQA 评测数据缺少 `ctxs` 字段 —— always_retrieve 退化为 no_retrieval

**问题描述**：
运行 `always_retrieve` 模式评测 PopQA 后，结果与 `no_retrieval` 完全相同（100% match，预测文本逐字一致）。

**根因分析**：
`data/eval/popqa_longtail_w_gs.jsonl` 文件中**不包含 `ctxs` 字段**（检索段落）。文件名中的 `_w_gs` 代表 "with gold standard answers"，而非 "with gold standard passages"。

`run_eval_batch_retrieve.py` 在第 193 行有 fallback 逻辑：
```python
if len(evidences) == 0:
    # 无检索段落 → 退回 no_retrieval
    all_prompts.append(prompt + "[No Retrieval]")
```

日志中的关键证据：
- `[2/5] Prompt 构建完成: 14267 samples × ~5 docs = 14267 prompts`（应为 71,335）
- `identical preds: 14267/14267`（与 no_retrieval 结果完全相同）

**Self-RAG 官方设计**：
论文评测使用的数据文件应包含 Contriever 预检索的 top-20 段落：
```json
{"question": "...", "answers": [...], "ctxs": [
  {"title": "...", "text": "...", "score": 0.85},
  ...
]}
```

这些数据原本通过 Google Drive 链接分发（已失效，见 P8）。

**解决方案**：
从 HuggingFace `selfrag/selfrag_train_data` 仓库的 `eval_data/` 目录下载包含 `ctxs` 字段的完整评测数据文件：

```bash
python -c "
from huggingface_hub import hf_hub_download
import shutil
fp = hf_hub_download(
    repo_id='selfrag/selfrag_train_data',
    filename='eval_data/popqa_longtail_w_gs.jsonl',
    repo_type='dataset',
)
shutil.copy2(fp, 'data/eval/popqa_longtail_w_gs.jsonl')
"
```

**工程经验**：
- 数据完整性是评测管道的第一道防线：应在推理前验证必需字段（`ctxs`）是否存在
- fallback 逻辑虽然防止了崩溃，但也可能**静默掩盖**数据问题——建议在 fallback 时增加明显的 WARNING
- `_w_gs` 这类缩写容易误导，应查看实际数据内容而非依赖文件名推断

---

## 附录：环境速查

```
集群:         Tang-2-Wu (8×NVIDIA A40, 48GB VRAM each)
项目路径:      /NAS/yesh/NLP/
Conda 环境:   /NAS/yesh/NLP/.conda/selfrag (Python 3.10)
HF 缓存:      /NAS/yesh/hf_cache/
模型目录:      /NAS/yesh/NLP/models/ (符号链接)
激活命令:      source /NAS/yesh/NLP/activate.sh
GitHub:       https://github.com/SunTomb/NLP.git
```

---

*最后更新：2026-04-26*

