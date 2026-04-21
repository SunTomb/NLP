"""
训练 Loss 曲线绘制脚本
用途：从 TensorBoard 日志或训练日志文件中提取 loss，绘制曲线

执行方式：
  python scripts/plot_loss.py --log_file logs/critic_train.log --output figures/critic_loss.png
  python scripts/plot_loss.py --tb_dir outputs/generator_llama2_7b --output figures/generator_loss.png
"""

import argparse
import re
import os
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# 设置图表风格
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 300,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


def parse_log_file(log_file):
    """从训练日志中提取 step 和 loss"""
    steps, losses = [], []
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # 匹配模式: Step: xxx, ... Loss: xxx
            match = re.search(r"Step:\s*(\d+).*Loss:\s*([\d.]+)", line)
            if match:
                steps.append(int(match.group(1)))
                losses.append(float(match.group(2)))
            # 匹配模式: {'loss': xxx, 'learning_rate': xxx, ...}
            match2 = re.search(r"'loss':\s*([\d.]+).*'epoch':\s*([\d.]+)", line)
            if match2 and not match:
                losses.append(float(match2.group(1)))
                steps.append(len(losses))
    return steps, losses


def parse_tensorboard(tb_dir):
    """从 TensorBoard 事件文件中提取 loss"""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(tb_dir)
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        loss_tag = None
        for tag in tags:
            if "loss" in tag.lower() and "train" in tag.lower():
                loss_tag = tag
                break
        if loss_tag is None:
            loss_tag = tags[0] if tags else None
        if loss_tag:
            events = ea.Scalars(loss_tag)
            steps = [e.step for e in events]
            losses = [e.value for e in events]
            return steps, losses
    except Exception as e:
        print(f"[WARN] TensorBoard 解析失败: {e}")
    return [], []


def plot_loss(steps, losses, title, output_file, smooth_window=10):
    """绘制 loss 曲线"""
    fig, ax = plt.subplots()

    # 原始 loss（浅色）
    ax.plot(steps, losses, alpha=0.3, color="#2196F3", linewidth=0.8, label="Raw Loss")

    # 平滑 loss（深色）
    if len(losses) > smooth_window:
        smoothed = np.convolve(losses, np.ones(smooth_window) / smooth_window, mode="valid")
        smooth_steps = steps[smooth_window - 1:]
        ax.plot(smooth_steps, smoothed, color="#1565C0", linewidth=2, label=f"Smoothed (window={smooth_window})")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()

    # 标注最终 loss
    if losses:
        final_loss = losses[-1]
        ax.annotate(
            f"Final: {final_loss:.4f}",
            xy=(steps[-1], final_loss),
            xytext=(-80, 20),
            textcoords="offset points",
            fontsize=10,
            arrowprops=dict(arrowstyle="->", color="gray"),
            color="#D32F2F",
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Loss 曲线已保存: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="绘制训练 Loss 曲线")
    parser.add_argument("--log_file", type=str, help="训练日志文件路径")
    parser.add_argument("--tb_dir", type=str, help="TensorBoard 日志目录")
    parser.add_argument("--output", type=str, default="figures/loss_curve.png", help="输出图片路径")
    parser.add_argument("--title", type=str, default="Training Loss", help="图表标题")
    parser.add_argument("--smooth", type=int, default=10, help="平滑窗口大小")
    args = parser.parse_args()

    steps, losses = [], []

    if args.log_file and os.path.exists(args.log_file):
        print(f"[INFO] 从日志文件提取: {args.log_file}")
        steps, losses = parse_log_file(args.log_file)
    elif args.tb_dir and os.path.exists(args.tb_dir):
        print(f"[INFO] 从 TensorBoard 提取: {args.tb_dir}")
        steps, losses = parse_tensorboard(args.tb_dir)

    if not losses:
        print("[ERROR] 未找到 loss 数据")
        return

    print(f"[INFO] 共 {len(losses)} 个数据点, Loss 范围: [{min(losses):.4f}, {max(losses):.4f}]")
    plot_loss(steps, losses, args.title, args.output, args.smooth)


if __name__ == "__main__":
    main()
