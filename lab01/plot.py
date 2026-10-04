"""从本地 CSV 导出训练、验证、测试曲线，不依赖 W&B 登录。"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator, PercentFormatter


def read_records(path, required_fields):
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not set(required_fields).issubset(reader.fieldnames or []):
            raise ValueError(f"{path.name} 缺少列：{required_fields}")
        records = []
        seen_epochs = set()
        for row in reader:
            epoch = int(row["epoch"])
            if epoch < 1 or epoch in seen_epochs:
                raise ValueError(f"{path.name} 的 epoch 必须为不重复的正整数")
            seen_epochs.add(epoch)
            record = {"epoch": epoch}
            for key in required_fields:
                if key == "epoch":
                    continue
                value = float(row[key])
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"{path.name} 第 {epoch} 轮的 {key} 无效")
                if key.endswith("_error") and value > 1:
                    raise ValueError(f"{key} 应记录 0 到 1 之间的比例")
                record[key] = value
            # 旧训练记录可能没有总损失，保留为空值。
            value = row.get("train_total_loss")
            record["train_total_loss"] = float(value) if value else math.nan
            records.append(record)
    if not records:
        raise ValueError(f"{path.name} 没有数据")
    return sorted(records, key=lambda record: record["epoch"])


def make_figure(history, test_records, name, loss_name):
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    font = next((name for name in ("Microsoft YaHei", "SimHei") if name in fonts), "DejaVu Sans")
    plt.rcParams.update({"font.family": font, "font.size": 11, "pdf.fonttype": 42})
    epochs = [record["epoch"] for record in history]
    test_by_epoch = {record["epoch"]: record for record in test_records}
    if set(test_by_epoch) - set(epochs):
        raise ValueError("测试记录包含训练记录中不存在的轮次，请重新运行 evaluate.py")

    fig, (loss_ax, error_ax) = plt.subplots(1, 2, figsize=(12, 5.2), layout="constrained")
    fig.suptitle(f"Fashion-MNIST | {name}", fontsize=15)
    colors = {"train": "#3266A8", "val": "#D07A28", "test": "#7B568C", "total": "#85732B"}

    loss_ax.plot(epochs, [row["train_loss"] for row in history], color=colors["train"],
                 marker="o", markersize=4, label="Train data loss")
    loss_ax.plot(epochs, [row["val_loss"] for row in history], color=colors["val"],
                 linestyle="--", marker="s", markersize=4, label="Validation data loss")
    if any(math.isfinite(row["train_total_loss"]) and
           abs(row["train_total_loss"] - row["train_loss"]) > 1e-10 for row in history):
        loss_ax.plot(epochs, [row["train_total_loss"] for row in history], color=colors["total"],
                     linestyle=":", label="Train objective (with L2)")
    loss_ax.set_title("Loss")
    loss_ax.set_ylabel(f"Loss ({loss_name})")
    loss_values = [row[key] for row in history
                   for key in ("train_loss", "val_loss", "train_total_loss")
                   if math.isfinite(row[key])]
    loss_ax.set_ylim(0, max(0.01, max(loss_values) * 1.1))

    for key, label, color, style, marker in (
        ("train_error", "Train", colors["train"], "-", "o"),
        ("val_error", "Validation", colors["val"], "--", "s"),
    ):
        error_ax.plot(epochs, [row[key] for row in history], label=label,
                      color=color, linestyle=style, marker=marker, markersize=4)
    if test_records:
        # 用 NaN 表示缺失轮次，避免连线暗示未测量的测试误差。
        errors = [test_by_epoch[epoch]["test_error"] if epoch in test_by_epoch else math.nan
                  for epoch in epochs]
        error_ax.plot(epochs, errors, label="Test", color=colors["test"],
                      linestyle="-.", marker="^", markersize=5)
    error_ax.set_title("Classification error")
    error_ax.set_ylabel("Error rate (%)")
    error_ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
    error_values = [row[key] for row in history for key in ("train_error", "val_error")]
    error_values.extend(row["test_error"] for row in test_records)
    error_ax.set_ylim(0, min(1.0, max(0.05, max(error_values) * 1.1)))
    best = min(history, key=lambda row: row["val_error"])
    error_ax.axvline(best["epoch"], color="#555555", linestyle=":", linewidth=1.2,
                    label=f"Best validation epoch: {best['epoch']}")
    missing = [epoch for epoch in epochs if epoch not in test_by_epoch]
    if missing:
        note = "Test evaluation not available" if not test_records else (
            f"Test coverage: {len(test_records)}/{len(epochs)} epochs; gaps left blank"
        )
        error_ax.text(0.02, 0.02, note, transform=error_ax.transAxes, fontsize=9, color="#555555")

    for ax in (loss_ax, error_ax):
        ax.set_xlabel("Epoch")
        ax.set_xlim(epochs[0] - 0.3, epochs[-1] + 0.3)
        ticks = MaxNLocator(integer=True, nbins=8).tick_values(epochs[0], epochs[-1])
        ax.set_xticks(sorted({epochs[0], epochs[-1]} | {
            tick for tick in ticks if epochs[0] <= tick <= epochs[-1]
        }))
        ax.grid(axis="y", color="#E3E5E8", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                  ncol=2, frameon=False, fontsize=9)
    return fig


def plot_run(run_dir):
    run_dir = Path(run_dir).resolve()
    history = read_records(run_dir / "history.csv",
                           ["epoch", "train_loss", "train_error", "val_loss", "val_error"])
    test_path = run_dir / "test_history.csv"
    summary_path = run_dir / "test_summary.json"
    if test_path.exists() and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        saved_digest = summary.get("train_history_sha256")
        current_digest = hashlib.sha256((run_dir / "history.csv").read_bytes()).hexdigest()
        if saved_digest is not None and saved_digest != current_digest:
            raise ValueError("训练记录已更新，测试评估已过期；请先重新运行 evaluate.py")
    test_records = read_records(test_path, ["epoch", "test_loss", "test_error"]) if test_path.exists() else []
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    fig = make_figure(history, test_records, config["name"], config.get("loss", "cross_entropy"))
    paths = [run_dir / "curves.png", run_dir / "curves.pdf"]
    try:
        for path in paths:
            fig.savefig(path, dpi=200, facecolor="white")
            print(f"已保存：{path}")
    finally:
        plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description="导出 Fashion-MNIST 实验的损失和误差曲线")
    parser.add_argument("--run", required=True, type=Path, help="实验目录，例如 results/baseline")
    args = parser.parse_args()
    try:
        plot_run(args.run)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
