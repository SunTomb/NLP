# Self-RAG 课程大作业 — 项目工作流 v3.0

> **更新日期**：2026 年 4 月 22 日 13:30  
> **基于**：PROJECT_WORKFLOW2.0.md  
> **本版重点**：Critic 全量训练已完成，进入 Generator 训练阶段

---

## 〇、总体进度

```
Phase 1: 基建        [███████████████████████████] 100%  ✅ 已完成
Phase 2: 复现实验    [██████████░░░░░░░░░░░░░░░░░]  40%  ◀ 当前阶段
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
| **M4** | **Critic 训练** | **4/22 12:40** | **3 epoch, Loss 13.47→0.22, 21.5h 单卡 A40** |

### M4 Critic 训练详细报告

| 指标 | 值 |
|------|-----|
| 数据量 | 48,500 条 |
| 总步数 | 18,186 步 |
| 训练时间 | 21 小时 35 分 |
| 平均速度 | 3.78 s/step, 1.87 samples/s |
| 最终 Loss | **0.22** |
| 配置 | 单卡 A40, bf16, Adafactor, GradCkpt, bs=1, accum=8 |

**Loss 曲线摘要**：

```
Step 20    → Loss 13.47  (初始，新 special token 未训练)
Step 60    → Loss  2.81  (急剧下降 -79%)
Step 200   → Loss  0.49  (进入精细学习)
Step 1760  → Loss  0.32  (Epoch 1 末尾)
Step 18186 → Loss  0.22  (最终收敛，总下降 98.4%)
```

### 遇到的工程问题汇总（详见 problem.md P9-P12）

| 编号 | 问题 | 影响 | 解决 |
|:----:|------|------|------|
| P9 | 相对导入失败 | 脚本无法执行 | 注释掉 flash_attn 导入 |
| P10 | PROMPT_DICT 缺键 | 数据加载失败 | 补充两个模板 |
| P11 | FSDP + GradCkpt 冲突 | 多卡 OOM | 改用单卡 |
| **P12** | **fp32 加载（根因）** | **所有 OOM** | **`torch_dtype=bf16`** |

---

## 二、下一步行动 — Generator 全量训练

> [!IMPORTANT]
> Generator 训练使用 `finetune.py`（基于 Accelerate），与 Critic 的 `train_special_tokens.py`（基于 HF Trainer）**架构完全不同**。

### 脚本差异对比

| 特性 | Critic (`train_special_tokens.py`) | Generator (`finetune.py`) |
|------|:----------------------------------:|:------------------------:|
| 训练框架 | HF Trainer | Accelerate (手写训练循环) |
| 优化器 | Adafactor (通过 --optim) | **AdamW (硬编码)** |
| Gradient Checkpointing | ✅ 支持 | ❌ 未实现 |
| 多卡支持 | FSDP | Accelerate DDP |
| 数据格式 | instruction+output | instruction+output (相同) |
| 模型加载 | ✅ 已修复 torch_dtype | ✅ 已修复 torch_dtype |
| 特殊Token | 15 个 | 15 个 (相同) |
| Max Seq Len | 512 | **2048** (更长！) |

### 显存预估（关键差异！）

> [!WARNING]
> Generator 训练使用 **AdamW** 优化器（不是 Adafactor），且序列长度 **2048**（不是 512）。显存需求远大于 Critic。

**单卡 A40 显存计算（AdamW + bf16 + seq=2048）**：

| 组件 | 大小 |
|------|:----:|
| 模型参数 (bf16) | 13.4 GB |
| AdamW 状态 (fp32 momentum + variance) | **26.8 GB** |
| 梯度 (bf16) | 13.4 GB |
| 激活值 (seq=2048, bs=1) | ~4-6 GB |
| **总计** | **~58-60 GB** |

> **结论**：单卡 A40 (46GB) **不够**。需要以下方案之一：

### 方案选择

| 方案 | 显存/卡 | 速度 | 推荐度 |
|------|:-------:|:----:|:------:|
| **A: 单卡 + Adafactor** | ~30GB | 慢 | ⭐⭐⭐ |
| B: 2卡 DDP + AdamW | ~58GB/卡 | ❌ OOM | ✗ |
| C: 4卡 FSDP + AdamW | ~20GB/卡 | 快 | ⭐⭐ |
| **D: 单卡 + LoRA + AdamW** | ~20GB | 最快 | ⭐⭐⭐⭐ |

> [!TIP]
> **推荐方案 A**（与 Critic 一致）：修改 `finetune.py` 将 AdamW 替换为 Adafactor，保持单卡训练。
> 
> **备选方案 D**（LoRA）：`finetune.py` 已内置 `--use_lora` 支持，显存最省，训练最快，但模型质量可能略低于全参数微调。建议先跑方案 A，如果时间紧张再用方案 D。

---

### Step 9: Generator 全量训练（方案 A：单卡 Adafactor）

#### 9.1 修改 finetune.py（在集群上执行）

需要将 AdamW 替换为 Adafactor 并启用 gradient checkpointing：

```bash
cd /NAS/yesh/NLP

python -c "
txt = open('self-rag/retrieval_lm/finetune.py').read()

# 1. 将 AdamW 替换为 Adafactor
old_optimizer = '''    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)'''
new_optimizer = '''    # 使用 Adafactor 替代 AdamW，节省 ~26GB 优化器显存（P12 经验）
    from transformers import Adafactor
    optimizer = Adafactor(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
    )'''
txt = txt.replace(old_optimizer, new_optimizer)

# 2. 在模型加载后启用 gradient_checkpointing
old_resize = '''    embedding_size = model.get_input_embeddings().weight.shape[0]'''
new_resize = '''    # 启用 gradient_checkpointing 节省激活值显存
    model.gradient_checkpointing_enable()
    print('gradient_checkpointing enabled ✅')

    embedding_size = model.get_input_embeddings().weight.shape[0]'''
txt = txt.replace(old_resize, new_resize)

# 3. 注释掉调试用的 processed.json 输出（145K 条数据会很慢）
txt = txt.replace(
    '''    with open(\"processed.json\", \"w\") as outfile:
        new_data = []
        for item in train_dataset:
            print(item)
            labels = [int(i) for i in item[\"labels\"]]
            input_ids = [int(i) for i in item[\"input_ids\"]]
            new_data.append({\"labels\": labels, \"input_ids\": input_ids})
        json.dump(new_data, outfile)''',
    '''    # 调试输出已禁用（145K 条数据会很慢）
    # with open(\"processed.json\", \"w\") as outfile:
    #     ...
    pass'''
)

open('self-rag/retrieval_lm/finetune.py', 'w').write(txt)
print('finetune.py 修改完成 ✅')
"
```

#### 9.2 确认 Generator 训练数据

```bash
# 确认数据文件
ls -la data/generator/
# 应该包含：output_selfrag_llama2_7b_0-145618_all.json 或类似文件

# 检查数据格式
python -c "
import json
with open('data/generator/output_selfrag_llama2_7b_0-145618_all.json') as f:
    data = json.load(f)
print(f'Generator 训练数据: {len(data)} 条')
print(f'数据字段: {list(data[0].keys())}')
print(f'示例 instruction: {data[0].get(\"instruction\", \"N/A\")[:100]}...')
print(f'示例 output 长度: {len(data[0].get(\"output\", \"\"))} chars')
"
```

#### 9.3 Generator 冒烟测试（先跑 100 条验证）

```bash
# 截取前 100 条做冒烟测试
python -c "
import json
with open('data/generator/output_selfrag_llama2_7b_0-145618_all.json') as f:
    data = json.load(f)
json.dump(data[:100], open('data/generator/generator_smoke_test.json', 'w'))
print(f'Generator 冒烟数据: 100 条')
"

# 冒烟测试（单卡 GPU 3，预计 5-10 分钟）
CUDA_VISIBLE_DEVICES=3 python self-rag/retrieval_lm/finetune.py \
    --model_name_or_path models/Llama-2-7b-hf \
    --train_file data/generator/generator_smoke_test.json \
    --use_special_tokens \
    --max_seq_length 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --num_train_epochs 1 \
    --output_dir outputs/generator_smoke \
    --logging_steps 5 \
    --low_cpu_mem_usage \
    --checkpointing_steps epoch \
    2>&1 | tee logs/generator_smoke_$(date +%Y%m%d_%H%M%S).log
```

#### 9.4 Generator 正式训练

> [!IMPORTANT]
> 冒烟测试通过后再执行正式训练。预计耗时 **2-3 天**（单卡 A40）。

```bash
# 在 tmux 中运行
tmux new -s gen_train

source /NAS/yesh/NLP/activate.sh
cd /NAS/yesh/NLP

# 确认空闲 GPU
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits

# 正式训练
CUDA_VISIBLE_DEVICES=3 python self-rag/retrieval_lm/finetune.py \
    --model_name_or_path models/Llama-2-7b-hf \
    --train_file data/generator/output_selfrag_llama2_7b_0-145618_all.json \
    --use_special_tokens \
    --max_seq_length 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --num_train_epochs 3 \
    --output_dir outputs/generator_llama2_7b \
    --logging_steps 50 \
    --low_cpu_mem_usage \
    --checkpointing_steps 5000 \
    2>&1 | tee logs/generator_train_$(date +%Y%m%d_%H%M%S).log

# 分离 tmux: Ctrl+B 然后按 D
```

**关键参数说明**：

| 参数 | 值 | 说明 |
|------|-----|------|
| `--max_seq_length 2048` | 2048 | Generator 输出包含完整检索段落+回答 |
| `--gradient_accumulation_steps 16` | 16 | 有效 batch = 16，平衡速度与显存 |
| `--warmup_ratio 0.03` | 3% | 145K × 3 epoch = 435K 样本，warmup ~13K 步 |
| `--checkpointing_steps 5000` | 5000 | 每 5000 步存一个 checkpoint |

**预计指标**：

| 指标 | 预估 |
|------|:----:|
| 总步数 | ~27,200 (145K ÷ 16 × 3) |
| 速度 | ~6-10 s/step (seq=2048 比 512 慢 3-4×) |
| 训练时间 | **2-3 天** |
| 显存 | ~28-35 GB |

#### 9.5 备选方案 D：LoRA 训练（如果时间紧张）

```bash
CUDA_VISIBLE_DEVICES=3 python self-rag/retrieval_lm/finetune.py \
    --model_name_or_path models/Llama-2-7b-hf \
    --train_file data/generator/output_selfrag_llama2_7b_0-145618_all.json \
    --use_special_tokens \
    --use_lora \
    --lora_rank 64 \
    --lora_alpha 16 \
    --max_seq_length 2048 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --num_train_epochs 3 \
    --output_dir outputs/generator_llama2_7b_lora \
    --logging_steps 50 \
    --low_cpu_mem_usage \
    --checkpointing_steps 5000 \
    2>&1 | tee logs/generator_lora_$(date +%Y%m%d_%H%M%S).log
```

LoRA 优势：显存仅 ~20GB，bs 可以开到 2，训练时间减半。

---

### Step 9 后验证

```bash
# 检查输出
ls -la outputs/generator_llama2_7b/

# 验证模型可加载 + 反思 Token 存在
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
model = AutoModelForCausalLM.from_pretrained('outputs/generator_llama2_7b', torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained('outputs/generator_llama2_7b')
print(f'Generator 模型加载成功 ✅')
print(f'  参数量: {sum(p.numel() for p in model.parameters())/1e9:.2f}B')
print(f'  词表: {len(tokenizer)}')
print(f'  Special tokens: {tokenizer.additional_special_tokens[:5]}')
"
```

---

## 三、后续步骤概览

### Step 10: 全面评测

在 Generator 训练完成后，评测我们复现的模型 vs 官方预训练 vs Vanilla Llama 2 基线。

```bash
# PopQA (no_retrieval 模式)
CUDA_VISIBLE_DEVICES=3 python self-rag/retrieval_lm/run_short_form.py \
    --model_name outputs/generator_llama2_7b \
    --input_file data/eval/popqa_longtail.jsonl \
    --max_new_tokens 100 --threshold 0.2 \
    --output_file results/popqa_our.json \
    --metric match --ndocs 0 --no_retrieval

# ARC-Challenge
CUDA_VISIBLE_DEVICES=3 python self-rag/retrieval_lm/run_short_form.py \
    --model_name outputs/generator_llama2_7b \
    --input_file data/eval/arc_challenge.jsonl \
    --max_new_tokens 50 --threshold 0.2 \
    --output_file results/arc_our.json \
    --metric match --ndocs 0 --no_retrieval
```

### Step 11-12: 改进实验

| 实验 | 基座 | 数据 | 预计时间 |
|------|------|------|:--------:|
| I1: Qwen 基座 | Qwen2.5-7B | 同 Generator 数据 | 2-3 天 |
| I2: 中文数据 | Qwen2.5-7B | 中文 QA + 反思 Token | 1-2 天 |

---

## 四、修订后时间线

```
Week:  W1(4/21)──W2(4/28)──W3(5/5)──W4(5/12)──W5(5/19)──W6(5/26)──W7(6/2)──W8(6/9)──→提交
       ├── P1: 基建 ─┤
       │ ✅ 全部完成 │
                     ├── P2: 复现训练 ───────────────┤
                     │ ✅ Critic (21.5h)             │
                     │ ◀ Generator (2-3d)            │
                     │   全面评测 (1d)                │
                     │   基线对比 (半天)              │
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
| **M5: Generator 冒烟** | **4/22** | **100 条无 crash** | **◀ 下一步** |
| M6: Generator 训练 | W3 5/1 | 3 epoch 完成 | 🔲 |
| M7: 复现评测 | W3-4 5/5 | 6 任务 + 基线对比 | 🔲 |
| M8: 改进实验 | W5-6 5/19 | Qwen / 中文 / 消融 | 🔲 |
| M9: 报告初稿 | W7 6/2 | ≥ 5000 字完整报告 | 🔲 |
| M10: 最终提交 | W8 6/15 | 报告 + 代码 + Demo | 🔲 |

---

## 六、训练前必检清单

- [ ] `nvidia-smi` 确认目标 GPU 空闲
- [ ] `tmux` 启动（防止 SSH 断连）
- [ ] `finetune.py` 已修改（Adafactor + GradCkpt + 禁用 processed.json）
- [ ] `torch_dtype=bf16` 已确认（第 442 行）
- [ ] Generator 数据文件存在且格式正确
- [ ] 先跑冒烟测试（100 条），通过后再正式训练

---

## 七、常用命令速查

```bash
# 环境
source /NAS/yesh/NLP/activate.sh

# GPU 状态（简洁版）
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits

# tmux
tmux new -s gen_train       # 创建
tmux attach -t gen_train    # 附加
# Ctrl+B D                  # 分离

# 监控训练
tail -f logs/generator_*.log
watch -n 60 nvidia-smi
```

---

## 八、文件清单

```
/NAS/yesh/NLP/
├── self-rag/
│   ├── data_creation/
│   │   └── train_special_tokens.py   # Critic 训练 ← 已修复并完成
│   └── retrieval_lm/
│       ├── finetune.py               # Generator 训练 ← 需修改后执行
│       ├── run_short_form.py         # Short-form 评测
│       └── run_long_form_static.py   # Long-form 评测
├── data/
│   ├── critic/
│   │   ├── critic_train_data_train.json  # 48.5K 条   ← 已用完
│   │   └── critic_smoke_test.json        # 100 条
│   └── generator/                        # 145K 条     ← 下一步使用
├── outputs/
│   ├── critic_smoke/              # 冒烟产出         ✅
│   ├── critic_llama2_7b/          # 正式 Critic      ✅
│   ├── generator_smoke/           # Generator 冒烟   ← 下一步
│   └── generator_llama2_7b/       # 正式 Generator   ← 后续
├── models/                        # 符号链接
│   ├── selfrag_llama2_7b → HF Cache
│   ├── Llama-2-7b-hf → HF Cache
│   ├── contriever-msmarco → HF Cache    # 评测检索器（待下载）
│   └── Qwen2.5-7B → HF Cache            # 改进实验（待下载）
└── logs/
    ├── critic_smoke_*.log          # 冒烟日志 (5 个)
    ├── critic_train_*.log          # Critic 训练日志   ✅
    └── generator_*.log             # Generator 日志    ← 待生成
```

---

> **立即行动**：
> 1. 在集群上修改 `finetune.py`（步骤 9.1）
> 2. 运行 Generator 冒烟测试（步骤 9.3）
> 3. 通过后启动正式训练（步骤 9.4）
