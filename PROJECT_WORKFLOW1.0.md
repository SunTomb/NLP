# Self-RAG 课程大作业 — 项目工作流 v1.0

> **更新日期**：2026 年 4 月 21 日 02:00  
> **基于**：PROJECT_WORKFLOW.md（v0.9 规划版）  
> **目的**：记录实际进度、集群环境参数，并为后续每一步提供「复制即用」的命令

---

## 〇、集群环境备忘录

> [!IMPORTANT]
> 以下信息是从实际集群操作中确认的，所有脚本和命令均基于此环境。

| 项目 | 值 |
|------|-----|
| **集群主机** | `Tang-2-Wu`（SSH: `wujcan@Tang2`） |
| **GPU 配置** | 8× NVIDIA A40 48GB（非 A100） |
| **CUDA 版本** | 12.2（nvcc V12.2.91） |
| **驱动版本** | 535.129.03 |
| **项目根目录** | `/NAS/yesh/NLP/` |
| **Conda 路径** | `/NAS/yesh/miniconda3/bin/conda` |
| **Conda 环境** | `/NAS/yesh/NLP/.conda/selfrag`（`--prefix` 安装，确保独立性） |
| **HF 缓存** | `/NAS/yesh/hf_cache/hub` |
| **HF 镜像** | `https://hf-mirror.com` |
| **Conda 镜像** | USTC + 清华镜像（已在 `.condarc` 中配置） |
| **self-rag 代码** | `/NAS/yesh/NLP/self-rag/`（已 clone） |

### 模型管理策略

> [!IMPORTANT]
> **所有模型统一存储在 `/NAS/yesh/hf_cache/hub/`**，`NLP/models/` 目录下使用**符号链接**指向 HF Cache。
> 这样做的好处：① 避免重复存储节省磁盘；② 多项目（G-MSRA、NLP 等）共享模型；③ 与 HuggingFace 缓存机制兼容。

**HF Cache 已有模型清单**（截至 2026-04-21）：

| HF Cache 目录名 | 对应模型 | 大小 | 本项目用途 |
|---|---|---|---|
| `models--meta-llama--Llama-2-7b-hf` | Llama 2 7B 基座 | ~14GB | ✅ 复现训练基座（已有，直接用） |
| `models--selfrag--selfrag_llama2_7b` | Self-RAG 预训练 | ~14GB | ✅ 推理验证（已有，直接用） |
| `models--Qwen--Qwen2.5-7B-Instruct` | Qwen 2.5 7B Instruct | ~14GB | ⚠️ 可做基线对比，I2 改进需 base 版 |
| `models--sentence-transformers--all-...` | Sentence Transformer | ~0.5GB | ✅ 检索 embedding 备用 |

**符号链接创建方法**：

```bash
# 查找 snapshot 路径并创建链接
SNAPSHOT=$(ls -d /NAS/yesh/hf_cache/hub/models--<org>--<name>/snapshots/*/ | head -1)
ln -s "$SNAPSHOT" /NAS/yesh/NLP/models/<name>
```

**未来下载新模型的标准流程**：

```bash
# 1. 下载到 HF Cache（不指定 --local-dir）
huggingface-cli download <repo_id>
# 2. 创建符号链接到项目目录
SNAPSHOT=$(ls -d /NAS/yesh/hf_cache/hub/models--<org>--<name>/snapshots/*/ | head -1)
ln -s "$SNAPSHOT" /NAS/yesh/NLP/models/<local_name>
```

### 环境激活命令

每次 SSH 登录后，执行：

```bash
source /NAS/yesh/NLP/activate.sh
```

`activate.sh` 内容：

```bash
cd /NAS/yesh/NLP
eval "$(/NAS/yesh/miniconda3/bin/conda shell.bash hook)"
conda activate /NAS/yesh/NLP/.conda/selfrag
export PATH="/NAS/yesh/NLP/.conda/selfrag/bin:$PATH"
export HF_CACHE=/NAS/yesh/hf_cache/hub
export HF_HOME=/NAS/yesh/hf_cache
export HUGGINGFACE_HUB_CACHE=/NAS/yesh/hf_cache/hub
export HF_ENDPOINT=https://hf-mirror.com
export PROJECT_DIR=/NAS/yesh/NLP
export PYTHONPATH=/NAS/yesh/NLP
```

### 关键注意事项

> [!WARNING]
> **pip 安装路径问题**：conda 环境的 site-packages 权限可能受限，导致 `pip install` 自动回退到 `--user`（安装到 `/home/wujcan/.local/`）。
>
> **解决方案**：始终使用完整路径调用 pip：
>
> ```bash
> /NAS/yesh/NLP/.conda/selfrag/bin/pip install <package>
> ```
>
> 或确保 `export PATH="/NAS/yesh/NLP/.conda/selfrag/bin:$PATH"` 在 pip 前执行。

### GPU 使用情况（2026-04-21 00:48 快照）

| GPU | 状态 | 占用 |
|-----|------|------|
| 0 | 🔴 占用 | 2× python 进程（~17.7GB） |
| 1 | 🔴 占用 | 1× python 进程（~41.3GB） |
| 2 | 🟢 空闲 | — |
| 3 | 🟢 空闲 | — |
| 4 | 🟢 空闲 | — |
| 5 | 🔴 占用 | 1× VLLM::EngineCore（~42.6GB） |
| 6 | 🟢 空闲 | — |
| 7 | 🟢 空闲 | — |

> 使用前请先 `nvidia-smi` 确认实时状态，**避免使用他人正在使用的 GPU**。

---

## 一、当前进度总览

### ✅ 已完成

| # | 任务 | 完成时间 | 备注 |
|---|------|---------|------|
| 0.1 | PROMPTS.md 精读 | 4/20 | 15 个阶段提示词手册 |
| 0.2 | PROJECT_WORKFLOW.md 制订 | 4/20 | 8 周排期规划（v0.9） |
| 0.3 | 官方代码精读 | 4/20 | train_special_tokens.py, finetune.py, README.md |
| 0.4 | 依赖兼容性分析 | 4/20 | requirements.txt vs 现代版本评估 |
| 1.1 | 项目目录创建 | 4/21 00:48 | `/NAS/yesh/NLP/` 结构建立 |
| 1.2 | self-rag 仓库克隆 | 4/21 00:48 | 已存在于 `/NAS/yesh/NLP/self-rag/` |
| 1.3 | Conda 环境创建 | 4/21 01:17 | `--prefix /NAS/yesh/NLP/.conda/selfrag` Python 3.10 |
| 1.4 | **依赖安装** | 4/21 01:35 | PyTorch 2.4.0+cu121, vllm 0.5.5, transformers 4.44.0, deepspeed 0.14.4 等全部安装 |
| 1.5 | PyTorch + CUDA 验证 | 4/21 01:35 | `torch.cuda.is_available() = True` ✅ |
| 2.1 | **Self-RAG 预训练模型下载** | 4/21 01:55 | HF Cache 已有，已创建符号链接 ✅ |
| 2.2 | **Llama 2 7B 基座确认** | 4/21 02:22 | HF Cache 已有 `models--meta-llama--Llama-2-7b-hf`，无需重复下载 ✅ |
| 2.3 | **模型目录重组** | 4/21 02:22 | 删除重复副本，改用 HF Cache 符号链接，节省 ~28GB ✅ |

### 🔲 已创建但待执行的脚本

| 脚本 | 路径（集群） | 功能 |
|------|-------------|------|
| `setup_env.sh` | `/NAS/yesh/NLP/scripts/setup_env.sh` | 一键环境部署（已手动完成） |
| `download_models.sh` | `/NAS/yesh/NLP/scripts/download_models.sh` | 模型下载（部分完成） |
| `download_data.sh` | `/NAS/yesh/NLP/scripts/download_data.sh` | 数据下载 |
| `quick_inference.py` | `/NAS/yesh/NLP/scripts/quick_inference.py` | 快速推理验证（**下一步**） |
| `train_critic.sh` | `/NAS/yesh/NLP/scripts/train_critic.sh` | Critic 训练 |
| `train_generator.sh` | `/NAS/yesh/NLP/scripts/train_generator.sh` | Generator 训练 |
| `train_critic_llama3.sh` | `/NAS/yesh/NLP/scripts/train_critic_llama3.sh` | Llama 3 Critic 训练 |
| `train_generator_llama3.sh` | `/NAS/yesh/NLP/scripts/train_generator_llama3.sh` | Llama 3 Generator 训练 |
| `eval/run_all_eval.sh` | `/NAS/yesh/NLP/scripts/eval/run_all_eval.sh` | 6 任务全面评测 |
| `eval/run_baselines.sh` | `/NAS/yesh/NLP/scripts/eval/run_baselines.sh` | 基线模型评测 |
| `eval/summarize_results.py` | `/NAS/yesh/NLP/scripts/eval/summarize_results.py` | 结果汇总+论文对比 |
| `ablation/run_ablation.sh` | `/NAS/yesh/NLP/scripts/ablation/run_ablation.sh` | 消融实验批量运行 |
| `plot_loss.py` | `/NAS/yesh/NLP/scripts/plot_loss.py` | Loss 曲线绘制 |
| `visualize.py` | `/NAS/yesh/NLP/scripts/visualize.py` | 报告图表生成 |
| `demo/app.py` | `/NAS/yesh/NLP/demo/app.py` | Gradio Demo 系统 |

---

## 二、下一步操作指南（Step by Step）

### Step 4：快速推理验证 ⬅️ 当前位置

**目的**：用官方预训练 Self-RAG 模型验证推理流程，观察反思 Token 行为。

```bash
# SSH 登录后
source /NAS/yesh/NLP/activate.sh
cd /NAS/yesh/NLP

# 使用空闲 GPU 运行推理（先 nvidia-smi 检查哪些空闲）
CUDA_VISIBLE_DEVICES=2 python scripts/quick_inference.py \
    --model_path models/selfrag_llama2_7b \
    --output_file results/quick_inference_results.json
```

**预期输出**：

- 7 个测试场景的推理结果
- 每个结果包含反思 Token 分析（`[Retrieval]`/`[No Retrieval]`/`[Relevant]` 等）
- 结果保存到 `results/quick_inference_results.json`

**预期耗时**：2-3 分钟（模型加载 ~1 分钟，推理 ~1 分钟）

**验证标准**：

- ✅ 简单数学题（2+2）应触发 `[No Retrieval]`
- ✅ 知识问题（llama vs alpaca）应触发 `[Retrieval]`
- ✅ 有检索段落输入时应出现 `[Relevant]` + `[Fully supported]`

---

### Step 5：模型符号链接设置 ✅ 已完成

**目的**：将 HF Cache 中已有的模型链接到项目目录，避免重复存储。

> [!NOTE]
> Llama 2 7B 和 Self-RAG 模型均已在 HF Cache 中存在，无需重复下载。

```bash
source /NAS/yesh/NLP/activate.sh
cd /NAS/yesh/NLP

# 删除重复副本（如果存在），释放 ~28GB
rm -rf models/selfrag_llama2_7b
rm -rf models/Llama-2-7b-hf

# 创建符号链接：Self-RAG 预训练模型
SELFRAG_SNAP=$(ls -d /NAS/yesh/hf_cache/hub/models--selfrag--selfrag_llama2_7b/snapshots/*/ | head -1)
ln -s "$SELFRAG_SNAP" models/selfrag_llama2_7b

# 创建符号链接：Llama 2 7B 基座
LLAMA2_SNAP=$(ls -d /NAS/yesh/hf_cache/hub/models--meta-llama--Llama-2-7b-hf/snapshots/*/ | head -1)
ln -s "$LLAMA2_SNAP" models/Llama-2-7b-hf

# 验证
ls -la models/
python -c "
from transformers import AutoTokenizer
t1 = AutoTokenizer.from_pretrained('models/selfrag_llama2_7b')
print(f'Self-RAG vocab: {len(t1)}, special tokens: {t1.additional_special_tokens[:3]}')
t2 = AutoTokenizer.from_pretrained('models/Llama-2-7b-hf')
print(f'Llama 2 vocab: {len(t2)}')
print('所有模型链接验证通过 ✅')
"
```

**后续模型下载**（进入改进实验阶段时执行）：

```bash
# Llama 3.1 8B（改进实验 I1）
huggingface-cli download meta-llama/Llama-3.1-8B
LLAMA3_SNAP=$(ls -d /NAS/yesh/hf_cache/hub/models--meta-llama--Llama-3.1-8B/snapshots/*/ | head -1)
ln -s "$LLAMA3_SNAP" models/Llama-3.1-8B

# Qwen 2.5 7B Base（改进实验 I2，注意 HF Cache 中已有 Instruct 版）
huggingface-cli download Qwen/Qwen2.5-7B
QWEN_SNAP=$(ls -d /NAS/yesh/hf_cache/hub/models--Qwen--Qwen2.5-7B/snapshots/*/ | head -1)
ln -s "$QWEN_SNAP" models/Qwen2.5-7B
```

---

### Step 6：下载训练数据

```bash
source /NAS/yesh/NLP/activate.sh

# 方式 A：运行下载脚本
bash scripts/download_data.sh

# 方式 B：手动执行（如果脚本有问题）

# 6.1 Critic 训练数据（Google Drive）
gdown 1IN1XcIOYtRIGWITJ4LKRgfITT-uUwk_W -O data/critic/critic_train_data.json
# 如果 gdown 失败 → 在本地浏览器下载后 scp 传到集群

# 6.2 Generator 训练数据（HuggingFace）
huggingface-cli download selfrag/selfrag_train_data \
    --local-dir data/generator/ \
    --repo-type dataset \
    --local-dir-use-symlinks False

# 6.3 评测数据
gdown 1TLKhWjez63H4uBtgCxyoyJsZi-IMgnDb -O data/eval/eval_data.zip
cd data/eval && unzip -o eval_data.zip && cd /NAS/yesh/NLP
```

**数据校验**：

```bash
python -c "
import json, os, glob
# Critic
if os.path.exists('data/critic/critic_train_data.json'):
    d = json.load(open('data/critic/critic_train_data.json'))
    print(f'Critic 数据: {len(d) if isinstance(d, list) else \"dict\"} 条')
# Generator
for f in glob.glob('data/generator/**/*.jsonl', recursive=True):
    n = sum(1 for _ in open(f))
    print(f'Generator 数据: {os.path.basename(f)} ({n:,} 条)')
# Eval
evals = glob.glob('data/eval/**/*.jsonl', recursive=True)
print(f'评测数据文件数: {len(evals)}')
"
```

---

### Step 7：Critic 训练冒烟测试

**目的**：用小批量数据验证训练脚本能跑通，不卡不 crash。

```bash
source /NAS/yesh/NLP/activate.sh

# 截取前 100 条 Critic 数据做冒烟测试
python -c "
import json
data = json.load(open('data/critic/critic_train_data.json'))
json.dump(data[:100] if isinstance(data, list) else data, 
          open('data/critic/critic_smoke_test.json', 'w'))
print(f'冒烟测试数据: 100 条')
"

# 用 1 张 GPU 跑 1 个 epoch 冒烟测试
cd self-rag/data_creation

CUDA_VISIBLE_DEVICES=3 python train_special_tokens.py \
    --model_name_or_path /NAS/yesh/NLP/models/Llama-2-7b-hf \
    --data_path /NAS/yesh/NLP/data/critic/critic_smoke_test.json \
    --use_special_token True \
    --bf16 True \
    --output_dir /NAS/yesh/NLP/outputs/critic_smoke_test \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --learning_rate 2e-5 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --model_max_length 512

cd /NAS/yesh/NLP
```

**预期**：训练应在 5-10 分钟内完成，loss 应从 ~2-3 逐步下降。

---

### Step 8：Critic 全量训练

> [!WARNING]
> **硬件调整**：原计划使用 2×A100 80GB，实际集群为 A40 48GB。A40 显存足够（Critic 训练约 ~35GB/卡），使用 2×A40 FSDP 即可。

```bash
source /NAS/yesh/NLP/activate.sh

# 建议在 tmux 中运行（防止 SSH 断连）
tmux new -s critic_train

cd /NAS/yesh/NLP

# 调整 train_critic.sh 中的 GPU 序号为空闲的 GPU
CUDA_VISIBLE_DEVICES=2,3 bash scripts/train_critic.sh
```

**关键参数**（已在 `train_critic.sh` 中配置）：

| 参数 | 值 | 说明 |
|------|-----|------|
| GPU | 2× A40 | FSDP 并行 |
| Epochs | 3 | 官方设置 |
| LR | 2e-5 | 官方设置 |
| Batch/GPU | 1 | 显存限制 |
| Grad Accum | 8 | 有效 batch = 16 |
| Max Seq Len | 512 | Critic 输入较短 |
| 预计时间 | **8-12 小时** | |

**监控**：

```bash
# 在另一个 tmux 窗口或新 SSH 中
tail -f /NAS/yesh/NLP/logs/critic_train.log
watch -n 30 nvidia-smi
```

---

### Step 9：Generator 全量训练

> [!WARNING]
> **硬件调整**：原计划 8×A100，调整为 **4×A40**（使用 DeepSpeed ZeRO-3 卸载到 CPU 以补偿显存）。或使用 **5×A40**（GPU 2-4, 6-7）以增大并行度。

```bash
source /NAS/yesh/NLP/activate.sh
tmux new -s gen_train

cd /NAS/yesh/NLP

# 使用空闲 GPU（避开 0,1,5）
CUDA_VISIBLE_DEVICES=2,3,4,6 bash scripts/train_generator.sh
```

**需要调整 `train_generator.sh` 中的参数**：

```bash
NUM_GPUS=4          # 改为实际使用的 GPU 数
TOTAL_BATCH_SIZE=128
BATCH_SIZE_PER_GPU=1
GRADIENT_ACC_STEPS=32   # 128 / 4 / 1 = 32
```

**预计时间**：1.5-2.5 天（A40 vs A100 慢约 30-50%）

---

### Step 10：全面评测

```bash
source /NAS/yesh/NLP/activate.sh

# 评测我们复现的模型
CUDA_VISIBLE_DEVICES=2 bash scripts/eval/run_all_eval.sh \
    outputs/generator_llama2_7b our_7b

# 评测官方预训练模型（交叉验证）
CUDA_VISIBLE_DEVICES=3 bash scripts/eval/run_all_eval.sh \
    models/selfrag_llama2_7b official_7b

# 评测基线（Vanilla Llama 2 + Standard RAG）
CUDA_VISIBLE_DEVICES=4 bash scripts/eval/run_baselines.sh

# 汇总所有结果
python scripts/eval/summarize_results.py --results_dir results
```

---

## 三、Phase 2-4 排期（根据实际硬件调整）

```
Week:  W1(4/21)──W2(4/28)──W3(5/5)──W4(5/12)──W5(5/19)──W6(5/26)──W7(6/2)──W8(6/9)──→提交
       ├── P1: 基建 ─┤
       │  ✅ 环境部署  │
       │  ✅ 模型下载  │
       │  → 推理验证   │
       │  → 数据下载   │
       │  → 冒烟测试   │
                       ├── P2: 复现实验 ──────────┤
                       │  Critic 训练 (8-12h)     │
                       │  Generator 训练 (1.5-2d)  │
                       │  全面评测 (1d)             │
                       │  基线对比 (半天)            │
                                                   ├── P3: 改进实验 ──────┤
                                                   │  Llama 3 训练 (I1)   │
                                                   │  中文适配 (I2)       │
                                                   │  消融实验             │
                                                   │  Demo 系统            │
                                                                          ├── P4: 报告 ─┤
                                                                          │  撰写+图表   │
                                                                          │  审校+提交   │
                                                                                         └→ 提交
```

### GPU 分配方案（A40 × 8）

| 阶段 | GPU 2-4 | GPU 6-7 | 说明 |
|------|---------|---------|------|
| W1 后半 | 推理验证 (1 GPU) | 空闲 | Step 4 |
| W2 | Critic 训练 (2 GPU) | 冒烟测试 | Step 7-8 |
| W3 | Generator 训练 (4 GPU) | 空闲 | Step 9 |
| W4 | 评测 (1-2 GPU) | 基线评测 (1-2 GPU) | Step 10 |
| W5-6 | Llama 3 训练 (4 GPU) | 中文评测 (1-2 GPU) | 改进实验 |
| W7 | 消融实验 (2-4 GPU) | Demo 系统 (1 GPU) | 消融+Demo |

> [!NOTE]
> GPU 0, 1, 5 被其他项目（可能是 G-MSRA）占用。**请勿使用这三张卡**，使用前务必 `nvidia-smi` 确认。

---

## 四、算力与时间预估（A40 版）

| 任务 | GPU 需求 | 显存/卡 | A40 预计时间 | 对比 A100 |
|------|---------|---------|-------------|-----------|
| 模型下载 | — | — | 15-30 min | 相同 |
| 推理验证 | 1× A40 | ~16GB | 2-3 min | 相同 |
| Critic 训练 | 2× A40 | ~35GB | 8-12 h | A100: 6-8h |
| Generator 训练 (7B) | 4× A40 | ~40GB | 1.5-2.5 天 | A100×8: 1-1.5 天 |
| 评测 (6 任务) | 1× A40 | ~16GB | 4-8 h | 相同 |
| 基线评测 | 1× A40 | ~16GB | 4-6 h | 相同 |
| Llama 3 训练 | 4× A40 | ~42GB | 2-3 天 | A100×8: 1-2 天 |
| 消融实验 (全部) | 2× A40 | ~16GB | 1 天 | 相同 |

---

## 五、文件结构

```
/NAS/yesh/NLP/
├── activate.sh                 # 环境激活脚本
├── self-rag/                   # 官方代码仓库（不修改）
│   ├── data_creation/
│   │   └── train_special_tokens.py   # Critic 训练代码
│   └── retrieval_lm/
│       ├── finetune.py               # Generator 训练代码
│       ├── run_short_form.py         # Short-form 评测
│       ├── run_long_form_static.py   # Long-form 评测
│       └── stage3_no_offloading_accelerate.conf  # DeepSpeed 配置
│
├── scripts/                    # 我们编写的脚本
│   ├── setup_env.sh
│   ├── download_models.sh
│   ├── download_data.sh
│   ├── quick_inference.py      # Step 4: 推理验证
│   ├── train_critic.sh         # Step 8: Critic 训练
│   ├── train_generator.sh      # Step 9: Generator 训练
│   ├── train_critic_llama3.sh  # I1: Llama 3 Critic
│   ├── train_generator_llama3.sh  # I1: Llama 3 Generator
│   ├── plot_loss.py            # Loss 曲线绘制
│   ├── visualize.py            # 报告图表
│   ├── eval/
│   │   ├── run_all_eval.sh     # 6 任务评测
│   │   ├── run_baselines.sh    # 基线评测
│   │   └── summarize_results.py  # 结果汇总
│   └── ablation/
│       └── run_ablation.sh     # 消融实验
│
├── demo/
│   └── app.py                  # Gradio Demo 系统
│
├── models/                     # 模型符号链接目录（实际文件在 HF Cache）
│   ├── selfrag_llama2_7b/ → /NAS/yesh/hf_cache/hub/models--selfrag--*.../snapshots/.../
│   ├── Llama-2-7b-hf/    → /NAS/yesh/hf_cache/hub/models--meta-llama--*.../snapshots/.../
│   ├── Llama-3.1-8B/           # 待链接（改进实验 I1）
│   └── Qwen2.5-7B/             # 待链接（改进实验 I2）
│
├── data/
│   ├── critic/                 # Critic 训练数据（Step 6）
│   ├── generator/              # Generator 训练数据（Step 6）
│   ├── eval/                   # 评测数据（Step 6）
│   └── chinese/                # 中文数据（I2 改进）
│
├── outputs/                    # 训练产出模型
├── results/                    # 评测结果
├── figures/                    # 可视化图表
├── logs/                       # 训练日志
└── .conda/
    └── selfrag/                # Conda 环境（独立于其他项目）
```

---

## 六、里程碑检查清单（更新版）

| 里程碑 | 目标时间 | 判定标准 | 状态 |
|--------|---------|---------|:----:|
| M0: 项目初始化 | W1 4/21 | 环境创建 + 依赖安装 + GPU 确认 | ✅ |
| M1: 推理验证通过 | W1 4/21 | 预训练模型推理输出含正确反思 Token | 🔲 进行中 |
| M2: 数据就绪 | W1 4/22 | Critic + Generator + 评测数据全部就位 | 🔲 |
| M3: 冒烟测试通过 | W2 4/25 | Critic + Generator 小数据训练无 crash | 🔲 |
| M4: Critic 训练完成 | W2-3 4/28 | 反思 Token 预测准确率 ≥ 论文值 | 🔲 |
| M5: Generator 训练完成 | W3-4 5/5 | Generator 输出含正确反思 Token | 🔲 |
| M6: 复现结果完成 | W4-5 5/12 | 6 个评测任务结果+基线对比 | 🔲 |
| M7: 基座升级完成 | W5-6 5/19 | Llama 3 版全量评测结果 | 🔲 |
| M8: 消融+分析完成 | W6-7 5/26 | 消融表+分析图+Case Study | 🔲 |
| M9: 报告初稿 | W7 6/2 | ≥ 5000 字完整报告 | 🔲 |
| M10: 最终提交 | W8 6/15 | 报告+代码+Demo | 🔲 |

---

## 七、常用命令速查

### 环境

```bash
source /NAS/yesh/NLP/activate.sh     # 激活环境
nvidia-smi                            # 查看 GPU
tmux new -s <name>                    # 新建 tmux
tmux attach -t <name>                 # 连接 tmux
# Ctrl+B D                           # 分离 tmux
```

### 训练监控

```bash
tail -f logs/critic_train.log         # 实时查看日志
watch -n 30 nvidia-smi                # 30秒刷新 GPU 状态
tensorboard --logdir outputs/xxx --port 6006  # TensorBoard
# SSH 端口转发: ssh -L 6006:localhost:6006 wujcan@Tang2
```

### 模型下载（统一使用 HF Cache + 符号链接）

```bash
# 1. 下载到 HF Cache
huggingface-cli download <repo_id>
# 2. 创建符号链接
SNAP=$(ls -d /NAS/yesh/hf_cache/hub/models--<org>--<name>/snapshots/*/ | head -1)
ln -s "$SNAP" models/<local_name>
```

### 推理测试

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> python scripts/quick_inference.py --model_path <path>
```

---

## 八、问题排查手册

| 问题 | 症状 | 解决 |
|------|------|------|
| pip 装到 ~/.local | `Defaulting to user installation` | 用完整路径 `/NAS/yesh/NLP/.conda/selfrag/bin/pip` |
| flash-attn 编译超时 | `Building wheel` 卡住 >30min | `Ctrl+C` 跳过，训练去掉 `--use_flash_attn` |
| CUDA OOM | `CUDA out of memory` | 降低 `batch_size` 或增加 `gradient_accumulation_steps` |
| HuggingFace 下载慢/失败 | 网络超时 | 确认 `export HF_ENDPOINT=https://hf-mirror.com` |
| gdown 下载失败 | Google Drive 限制 | 本地浏览器下载后 `scp` 到集群 |
| vllm import 失败 | GPU 节点问题 | 确认在有 GPU 的节点运行，且 CUDA 版本匹配 |
| Llama 2 下载被拒 | 403 Forbidden | 在 HuggingFace 网站接受 Meta Llama 协议后重试 |
| train_special_tokens.py 相对导入错误 | `ImportError` | 在脚本开头添加 `sys.path` 或去掉 flash_attn 参数 |

---

> **下一步行动**：执行 Step 4（快速推理验证），确认 Self-RAG 模型工作正常后，进入数据下载和训练流程。
