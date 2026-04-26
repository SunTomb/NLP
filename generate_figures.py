"""
Self-RAG 实验报告图表生成脚本
生成所有实验结果的可视化图表
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import json, os, re

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.bbox'] = 'tight'

OUT = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUT, exist_ok=True)
RESULTS = os.path.join(os.path.dirname(__file__), 'results')

def load(fname):
    p = os.path.join(RESULTS, fname)
    if not os.path.exists(p): return None
    return json.load(open(p, encoding='utf-8'))

# ============================================================
# Fig 1: 多任务评测柱状图 (Our vs Official vs Llama2)
# ============================================================
def fig1_main_results():
    tasks = ['PopQA\n(no_ret)', 'PopQA\n(always_ret)', 'ARC-C', 'TriviaQA']
    our    = [23.45, 50.46, 57.25, 31.50]
    off    = [28.66, 52.32, 62.29, 29.75]
    llama  = [None,  None,  43.34, 17.05]

    x = np.arange(len(tasks))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5.5))

    bars1 = ax.bar(x - w, our, w, label='Our (Llama-2)', color='#3b82f6', edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x, off, w, label='Official', color='#10b981', edgecolor='white', linewidth=0.5)
    llama_vals = [v if v else 0 for v in llama]
    bars3 = ax.bar(x + w, llama_vals, w, label='Llama-2 (base)', color='#f59e0b', edgecolor='white', linewidth=0.5)
    # hide llama bars where None
    for i, v in enumerate(llama):
        if v is None:
            bars3[i].set_visible(False)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            if bar.get_visible() and bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                        f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('多任务评测结果对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=10)
    ax.set_ylim(0, 75)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig(os.path.join(OUT, 'fig1_main_results.pdf'))
    fig.savefig(os.path.join(OUT, 'fig1_main_results.png'))
    plt.close(fig)
    print('✅ fig1_main_results')

# ============================================================
# Fig 2: 检索增强效果对比 (no_ret vs always_ret)
# ============================================================
def fig2_retrieval_effect():
    models = ['Our (Llama-2)', 'Official']
    nr = [23.45, 28.66]
    ar = [50.46, 52.32]
    gain = [ar[i]-nr[i] for i in range(2)]

    x = np.arange(len(models))
    w = 0.3
    fig, ax = plt.subplots(figsize=(7, 5))

    b1 = ax.bar(x - w/2, nr, w, label='No Retrieval', color='#94a3b8', edgecolor='white')
    b2 = ax.bar(x + w/2, ar, w, label='Always Retrieve', color='#3b82f6', edgecolor='white')

    for i in range(len(models)):
        ax.annotate(f'+{gain[i]:.1f}pp', xy=(x[i]+w/2, ar[i]),
                    xytext=(x[i]+w/2+0.15, ar[i]+3), fontsize=11, fontweight='bold', color='#dc2626',
                    arrowprops=dict(arrowstyle='->', color='#dc2626', lw=1.5))
        ax.text(x[i]-w/2, nr[i]+0.8, f'{nr[i]:.1f}%', ha='center', fontsize=9, fontweight='bold')
        ax.text(x[i]+w/2, ar[i]+0.8, f'{ar[i]:.1f}%', ha='center', fontsize=9, fontweight='bold')

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('PopQA 检索增强效果', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 65)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig(os.path.join(OUT, 'fig2_retrieval_effect.pdf'))
    fig.savefig(os.path.join(OUT, 'fig2_retrieval_effect.png'))
    plt.close(fig)
    print('✅ fig2_retrieval_effect')

# ============================================================
# Fig 3: 消融实验柱状图
# ============================================================
def fig3_ablation():
    labels = ['Full\n(G+U)', 'w/o\nGroundness', 'w/o\nUtility', 'w/o\nAll Scoring']
    vals = [50.46, 46.25, 50.54, 46.32]
    colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, vals, color=colors, edgecolor='white', linewidth=0.5, width=0.55)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.5, f'{v:.2f}%', ha='center', fontsize=10, fontweight='bold')

    # 标注差值
    ax.annotate('−4.22pp', xy=(1, 46.25), xytext=(1, 42), fontsize=10, color='#dc2626',
                fontweight='bold', ha='center', arrowprops=dict(arrowstyle='->', color='#dc2626'))
    ax.annotate('+0.07pp', xy=(2, 50.54), xytext=(2, 53.5), fontsize=10, color='#16a34a',
                fontweight='bold', ha='center')
    ax.annotate('−4.15pp', xy=(3, 46.32), xytext=(3, 42), fontsize=10, color='#dc2626',
                fontweight='bold', ha='center', arrowprops=dict(arrowstyle='->', color='#dc2626'))

    ax.axhline(y=50.46, color='#3b82f6', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('反思 Token 消融实验 (PopQA always_retrieve)', fontsize=13, fontweight='bold')
    ax.set_ylim(40, 56)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig(os.path.join(OUT, 'fig3_ablation.pdf'))
    fig.savefig(os.path.join(OUT, 'fig3_ablation.png'))
    plt.close(fig)
    print('✅ fig3_ablation')

# ============================================================
# Fig 4: Generator Loss 曲线 (从日志解析)
# ============================================================
def parse_loss(logfile):
    steps, losses = [], []
    if not os.path.exists(logfile): return steps, losses
    with open(logfile, encoding='utf-8', errors='replace') as f:
        for line in f:
            if 'Step:' in line and 'Loss:' in line:
                try:
                    s = int(line.split('Step:')[1].split(',')[0].strip())
                    l = float(line.split('Loss:')[1].strip())
                    steps.append(s)
                    losses.append(l)
                except: pass
    return steps, losses

def fig4_loss_curves():
    logs_dir = os.path.join(os.path.dirname(__file__), 'logs')

    # Generator Llama-2: 使用最大的 gen_train 日志
    gen_log = os.path.join(logs_dir, 'gen_train_20260423_010758.log')
    if not os.path.exists(gen_log): gen_log = None
    # Qwen
    qwen_log = os.path.join(logs_dir, 'train_qwen_20260426_072238.log')
    if not os.path.exists(qwen_log): qwen_log = None
    # ZH
    zh_log = os.path.join(logs_dir, 'train_zh_20260426_072238.log')
    if not os.path.exists(zh_log): zh_log = None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Generator Llama-2 vs Qwen
    if gen_log:
        s, l = parse_loss(gen_log)
        if s:
            # 降采样
            idx = list(range(0, len(s), max(1, len(s)//500)))
            axes[0].plot([s[i] for i in idx], [l[i] for i in idx], alpha=0.7, linewidth=0.8, label='Llama-2-7B', color='#3b82f6')
    if qwen_log:
        s, l = parse_loss(qwen_log)
        if s:
            idx = list(range(0, len(s), max(1, len(s)//500)))
            axes[0].plot([s[i] for i in idx], [l[i] for i in idx], alpha=0.7, linewidth=0.8, label='Qwen2.5-7B', color='#ef4444')

    axes[0].set_xlabel('Training Step', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].set_title('Generator 训练 Loss 曲线', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].set_ylim(0, 7)
    axes[0].grid(alpha=0.3)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # Right: ZH 训练
    if zh_log:
        s, l = parse_loss(zh_log)
        if s:
            axes[1].plot(s, l, color='#8b5cf6', linewidth=1.5, marker='o', markersize=2, alpha=0.7)
    axes[1].set_xlabel('Training Step', fontsize=11)
    axes[1].set_ylabel('Loss', fontsize=11)
    axes[1].set_title('中文适配 (Qwen2.5) 训练 Loss', fontsize=13, fontweight='bold')
    axes[1].grid(alpha=0.3)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig4_loss_curves.pdf'))
    fig.savefig(os.path.join(OUT, 'fig4_loss_curves.png'))
    plt.close(fig)
    print('✅ fig4_loss_curves')

# ============================================================
# Fig 5: Self-RAG 提升幅度 (相对 Llama-2 基线)
# ============================================================
def fig5_improvement():
    tasks = ['ARC-C', 'TriviaQA']
    llama = [43.34, 17.05]
    our   = [57.25, 31.50]
    off   = [62.29, 29.75]
    gain_our = [our[i]-llama[i] for i in range(2)]
    gain_off = [off[i]-llama[i] for i in range(2)]

    x = np.arange(len(tasks))
    w = 0.3
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.bar(x - w, llama, w, label='Llama-2 (base)', color='#94a3b8', edgecolor='white')
    b2 = ax.bar(x, our, w, label='Our (Self-RAG)', color='#3b82f6', edgecolor='white')
    b3 = ax.bar(x + w, off, w, label='Official', color='#10b981', edgecolor='white')

    for i in range(2):
        ax.annotate(f'+{gain_our[i]:.1f}pp', xy=(x[i], our[i]),
                    xytext=(x[i], our[i]+2.5), fontsize=10, fontweight='bold',
                    ha='center', color='#3b82f6')

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Self-RAG 微调提升幅度 (相对 Llama-2 基线)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=11)
    ax.set_ylim(0, 75)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig(os.path.join(OUT, 'fig5_improvement.pdf'))
    fig.savefig(os.path.join(OUT, 'fig5_improvement.png'))
    plt.close(fig)
    print('✅ fig5_improvement')

# ============================================================
if __name__ == '__main__':
    print('开始生成图表...\n')
    fig1_main_results()
    fig2_retrieval_effect()
    fig3_ablation()
    fig4_loss_curves()
    fig5_improvement()
    print(f'\n所有图表已保存到 {OUT}/')
