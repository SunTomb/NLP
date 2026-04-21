# Self-RAG 课程大作业 — 项目工作流 v2.0

> **更新日期**：2026 年 4 月 21 日 15:00  
> **基于**：PROJECT_WORKFLOW1.0.md  
> **本版重点**：冒烟训练已通过，进入正式 Critic 训练阶段

---

## 〇、总体进度

```
Phase 1: 基建        [███████████████████████████] 100%  ✅ 已完成
Phase 2: 复现实验    [████░░░░░░░░░░░░░░░░░░░░░░]  15%  ◀ 当前阶段
Phase 3: 改进实验    [░░░░░░░░░░░░░░░░░░░░░░░░░░]   0%
Phase 4: 报告撰写    [░░░░░░░░░░░░░░░░░░░░░░░░░░]   0%
```

---

## 一、已完成里程碑

| # | 里程碑 | 完成时间 | 关键结果 |
|---|--------|---------|---------|
| M0 | 环境部署 | 4/21 01:35 | conda --prefix 隔离环境，PyTorch 2.4 + vLLM 0.5.5 |
| M1 | 推理验证 | 4/21 02:30 | 7 组测试全部通过，反思 Token 行为符合论文 |
| M2 | 数据就绪 | 4/21 ~04:00 | Critic 48.5K 条合成 + Generator 145K 条 + 4 个评测集 |
| M3 | 冒烟测试 | 4/21 14:50 | **100 条数据，Loss 8.43→5.16，单卡 A40 ~22GB** |

### M3 冒烟训练关键指标

```
总步数:   25 步 (100条 ÷ batch1 ÷ grad_accum4)
训练时间: 63 秒
吞吐量:   1.58 samples/s, 2.53 s/step
显存:     ~22 GB (单卡 A40, bf16 + Adafactor + GradCkpt)
```

| Epoch | Loss | Grad Norm | 分析 |
|:-----:|:----:|:---------:|------|
| 0.2 | 8.43 | 1064.0 | 新 special tokens 随机初始化，梯度大 |
| 0.4 | 7.09 | 30.0 | 梯度迅速稳定 |
| 0.6 | 5.60 | 27.75 | 持续下降 |
| 0.8 | 5.18 | 35.5 | 趋于收敛 |
| 1.0 | 5.16 | 166.0 | lr→0 附近的正常波动 |

### 遇到的工程问题（详见 problem.md）

| 编号 | 问题 | 解决 |
|:----:|------|------|
| P9 | 相对导入失败 | 注释掉 flash_attn 导入 |
| P10 | PROMPT_DICT 缺键 | 补充两个模板 |
| P11 | FSDP + GradCkpt 冲突 | 改用单卡 / 正确 FSDP 配置 |
| **P12** | **fp32 加载（根因）** | **`from_pretrained` 加 `torch_dtype=bf16`** |

---

## 二、下一步行动 — Critic 全量训练

### Step 8: Critic 全量训练

> [!IMPORTANT]
> 本步骤预计耗时 **3-6 小时**（单卡 A40, Adafactor），请在 tmux 中运行。

**训练配置**：

| 参数 | 值 | 说明 |
|------|-----|------|
| 数据 | `data/critic/critic_train_data_train.json` | 48,500 条 |
| 验证集 | `data/critic/critic_train_data_dev.json` | ~ 5K 条 |
| 基座模型 | `models/Llama-2-7b-hf` | NousResearch 镜像 |
| GPU | 单卡 A40 (48GB) | 预计占用 ~22-25 GB |
| 精度 | bf16 | 模型 + 训练均使用 bf16 |
| 优化器 | Adafactor | 无 momentum，极低显存 |
| Epochs | 3 | 官方设置 |
| LR | 2e-5 | cosine scheduler |
| Batch/GPU | 1 | 显存安全 |
| Grad Accum | 8 | 有效 batch = 8 |
| Max Seq Len | 512 | Critic 输入较短 |
| Checkpointing | gradient_checkpointing | 单卡无 FSDP 冲突 |

**执行命令**：

```bash
# 1. SSH 登录后激活环境
source /NAS/yesh/NLP/activate.sh
cd /NAS/yesh/NLP

# 2. 启动 tmux（防止 SSH 断连导致训练中断）
tmux new -s critic_train

# 3. 确认空闲 GPU
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits
# 选择 memory.used 最小的那张卡，假设是 GPU 3

# 4. 启动正式训练
CUDA_VISIBLE_DEVICES=3 python self-rag/data_creation/train_special_tokens.py \
    --model_name_or_path models/Llama-2-7b-hf \
    --data_path data/critic/critic_train_data_train.json \
    --output_dir outputs/critic_llama2_7b \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type cosine \
    --bf16 True \
    --model_max_length 512 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --logging_steps 20 \
    --gradient_checkpointing True \
    --optim adafactor \
    --use_special_token True \
    --report_to none \
    2>&1 | tee logs/critic_train_$(date +%Y%m%d_%H%M%S).log

# 5. 分离 tmux（Ctrl+B 然后按 D）
```

**多卡训练方案（可选，如需加速）**：

如果有 4 张空闲卡（完全空闲，memory.used ≈ 0），可用 FSDP 加速：

```bash
# 注意：FSDP 模式下不要使用 --gradient_checkpointing，改用 fsdp_config
CUDA_VISIBLE_DEVICES=1,3,5,6 torchrun --nproc_per_node=4 --master_port=29500 \
    self-rag/data_creation/train_special_tokens.py \
    --model_name_or_path models/Llama-2-7b-hf \
    --data_path data/critic/critic_train_data_train.json \
    --output_dir outputs/critic_llama2_7b \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type cosine \
    --bf16 True \
    --model_max_length 512 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --logging_steps 20 \
    --optim adafactor \
    --use_special_token True \
    --fsdp "full_shard auto_wrap" \
    --report_to none \
    2>&1 | tee logs/critic_train_$(date +%Y%m%d_%H%M%S).log
```

> [!WARNING]
> FSDP 模式下**不要**同时使用 `--gradient_checkpointing True`（见 P11）。
> 如需 activation checkpointing，请通过 `--fsdp_config` 传入。

### 监控命令

```bash
# 在另一个 tmux 窗口或新 SSH 中
tail -f logs/critic_train_*.log | grep -E 'loss|eval|save'  # 实时 loss
watch -n 60 nvidia-smi                                       # GPU 监控
```

### 训练完成后验证

```bash
# 检查输出目录
ls -la outputs/critic_llama2_7b/

# 验证模型可加载
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
model = AutoModelForCausalLM.from_pretrained('outputs/critic_llama2_7b', torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained('outputs/critic_llama2_7b')
print(f'Critic 模型加载成功 ✅')
print(f'  参数量: {sum(p.numel() for p in model.parameters())/1e9:.2f}B')
print(f'  词表大小: {len(tokenizer)} (应含 15 个 special tokens)')
print(f'  Special tokens: {tokenizer.additional_special_tokens[:5]}')
"
```

### 预期结果

- Loss 应从 ~5-8 持续下降到 ~2-3（3 个 epoch）
- 训练完成后 `outputs/critic_llama2_7b/` 包含完整模型文件
- 词表应包含 15 个反思 Token

---

## 三、后续步骤概览

### Step 9: Generator 全量训练

> 在 Critic 训练完成后执行

**配置**：

| 参数 | 值 |
|------|-----|
| 数据 | `data/generator/output_selfrag_llama2_7b_0-145618_all.json` |
| 基座 | `models/Llama-2-7b-hf` |
| GPU | 4× A40 (FSDP 或 DeepSpeed ZeRO-3) |
| Epochs | 3 |
| 有效 Batch | 128 (batch=1 × accum=32 × 4 GPUs) |
| 预计时间 | 1.5-2.5 天 |
| 优化器 | AdamW 或 Adafactor |

**训练脚本**（使用官方的 `finetune.py`）：

```bash
# 同样需要修复 from_pretrained 的 torch_dtype 问题
# finetune.py 位于 self-rag/retrieval_lm/finetune.py

CUDA_VISIBLE_DEVICES=0,2,4,6 torchrun --nproc_per_node=4 --master_port=29501 \
    self-rag/retrieval_lm/finetune.py \
    --model_name_or_path models/Llama-2-7b-hf \
    --data_path data/generator/output_selfrag_llama2_7b_0-145618_all.json \
    --output_dir outputs/generator_llama2_7b \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --learning_rate 2e-5 \
    --bf16 True \
    --model_max_length 2048 \
    --fsdp "full_shard auto_wrap" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --logging_steps 50 \
    --report_to none \
    2>&1 | tee logs/generator_train_$(date +%Y%m%d_%H%M%S).log
```

> [!WARNING]
> `finetune.py` 可能也有 `from_pretrained` 不指定 `torch_dtype` 的问题，执行前需要先检查并修复（同 P12）。
> Generator 的 `model_max_length=2048` 比 Critic 长很多，显存需求更大。

---

### Step 10: 全面评测

评测 6 个任务：PopQA、ARC-Challenge、PubHealth、TriviaQA，对比以下模型：

| 模型 | 来源 |
|------|------|
| 官方 Self-RAG | `models/selfrag_llama2_7b` |
| 我们复现的 Generator | `outputs/generator_llama2_7b` |
| Vanilla Llama 2 (基线) | `models/Llama-2-7b-hf` |

```bash
# 以 PopQA 为例（no_retrieval 模式）
CUDA_VISIBLE_DEVICES=3 python self-rag/retrieval_lm/run_short_form.py \
    --model_name outputs/generator_llama2_7b \
    --input_file data/eval/popqa_longtail.jsonl \
    --max_new_tokens 100 \
    --threshold 0.2 \
    --output_file results/popqa_our_model.json \
    --metric match \
    --ndocs 0 \
    --no_retrieval
```

> [!NOTE]
> 我们的评测数据不包含预检索段落。若需要带检索的评测，需先运行 Contriever-MSMARCO 检索器为每条数据添加上下文段落。`no_retrieval` 模式可以先跑起来对比。

---

## 四、修订后时间线

```
Week:  W1(4/21)──W2(4/28)──W3(5/5)──W4(5/12)──W5(5/19)──W6(5/26)──W7(6/2)──W8(6/9)──→提交
       ├── P1: 基建 ─┤
       │ ✅ 环境    │
       │ ✅ 模型    │
       │ ✅ 推理    │
       │ ✅ 数据    │
       │ ✅ 冒烟    │
                     ├── P2: 复现训练 ───────────────┤
                     │ ◀ Critic 训练 (3-6h)         │
                     │   Generator 训练 (1.5-2.5d)   │
                     │   全面评测 (1d)                │
                     │   基线对比 (半天)              │
                                                     ├── P3: 改进实验 ──────┤
                                                     │  Llama 3 (I1)       │
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
| M1: 推理验证 | 4/21 | 反思 Token 行为正确 | ✅ |
| M2: 数据就绪 | 4/21 | Critic + Generator + Eval | ✅ |
| M3: 冒烟通过 | 4/21 | Loss 正常下降，无 crash | ✅ |
| M4: Critic 训练 | W2 4/25 | 3 epoch 完成，loss 收敛 | ◀ 下一步 |
| M5: Generator 训练 | W3 5/5 | 输出含正确反思 Token | 🔲 |
| M6: 复现评测 | W4 5/12 | 6 任务 + 基线对比 | 🔲 |
| M7: 改进实验 | W5-6 5/19 | Llama 3 / 中文 / 消融 | 🔲 |
| M8: 报告初稿 | W7 6/2 | ≥ 5000 字完整报告 | 🔲 |
| M9: 最终提交 | W8 6/15 | 报告 + 代码 + Demo | 🔲 |

---

## 六、关键路径与注意事项

### 已验证的最佳训练配置（Critic）

```
✅ 单卡 A40 + bf16 + Adafactor + gradient_checkpointing
   显存: ~22 GB / 46 GB
   速度: 2.53 s/step
```

### 训练前必检清单

1. **`nvidia-smi`** — 确认目标 GPU 完全空闲（memory.used ≈ 0）
2. **`tmux`** — 防止 SSH 断连中断训练
3. **`torch_dtype=bf16`** — 确认 `train_special_tokens.py` 已修复 P12
4. **日志保存** — 所有训练都用 `| tee logs/xxx.log` 保存

### 常用命令速查

```bash
# 环境
source /NAS/yesh/NLP/activate.sh

# GPU 状态
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits

# tmux
tmux new -s <name>         # 创建
tmux attach -t <name>      # 附加
# Ctrl+B D                 # 分离

# 监控训练
tail -f logs/*.log | grep loss
watch -n 60 nvidia-smi

# Git 同步
git add -A && git commit -m "update" && git push
```

---

## 七、文件清单更新

```
/NAS/yesh/NLP/
├── activate.sh                    # 环境激活
├── problem.md                     # 问题记录 (P1-P12)          ← 已更新
├── PROJECT_WORKFLOW1.0.md         # 旧版工作流
├── PROJECT_WORKFLOW2.0.md         # 本文件                     ← 新增
├── self-rag/                      # 官方代码（已修复 P9/P10/P12）
│   ├── data_creation/
│   │   └── train_special_tokens.py   # Critic 训练 ← 已修复
│   └── retrieval_lm/
│       ├── finetune.py               # Generator 训练 ← 待检查 P12
│       └── run_short_form.py         # 评测
├── scripts/
│   ├── prepare_data.py            # 数据合成/下载工具
│   └── quick_inference.py         # 推理验证
├── models/                        # 符号链接 → HF Cache
│   ├── selfrag_llama2_7b → ...
│   └── Llama-2-7b-hf → ...
├── data/
│   ├── critic/
│   │   ├── critic_train_data_train.json  # 48,500 条
│   │   ├── critic_train_data_dev.json    # 验证集
│   │   └── critic_smoke_test.json        # 100 条冒烟数据
│   ├── generator/                        # 145K 条
│   └── eval/                             # PopQA, ARC, PubHealth, TriviaQA
├── outputs/
│   ├── critic_smoke/              # 冒烟训练产出     ← 已完成
│   ├── critic_llama2_7b/          # 正式 Critic      ← 下一步
│   └── generator_llama2_7b/       # 正式 Generator   ← 后续
├── results/                       # 评测结果
└── logs/                          # 训练日志
    ├── critic_smoke_*.log         # 冒烟测试日志 (5 个)
    └── critic_train_*.log         # 正式训练日志 ← 待生成
```

---

> **立即行动**：在集群上执行 Step 8（Critic 全量训练），确认 `train_special_tokens.py` 已包含 `torch_dtype` 修复后，启动 tmux 训练。
