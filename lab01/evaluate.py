"""实验方案确定后，评估每轮模型和验证集选出的 best.pth。"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

from data import get_test_loader
from engine import evaluate as evaluate_model
from losses import build_loss
from model import MLP


def normalized_config(config):
    config = dict(config)
    # 兼容加入损失函数选项、L2、Dropout 和 BN 之前的验证集实验。
    for key, default in (
        ("loss", "cross_entropy"), ("l2", 0.0),
        ("dropout", 0.0), ("batch_norm", False),
    ):
        config.setdefault(key, default)
    return config


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def checkpoint_epoch(checkpoint):
    epoch = checkpoint.get("epoch", checkpoint.get("completed_epochs"))
    if not isinstance(epoch, int) or epoch < 1:
        raise ValueError("检查点缺少有效的 epoch 或 completed_epochs")
    return epoch


def evaluate_run(run_dir, device="auto", batch_size=None):
    run_dir = Path(run_dir).resolve()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("当前环境无法使用 CUDA，请指定 --device cpu")
    device = torch.device(device)

    last = load_checkpoint(run_dir / "last.pth")
    best = load_checkpoint(run_dir / "best.pth")
    config = normalized_config(last["config"])
    if config.get("data_protocol") != "fashion_stratified_50k_10k_v1":
        raise ValueError("该实验未使用当前验证集划分，请使用新的验证集实验")
    completed_epochs = checkpoint_epoch(last)
    training_history_path = run_dir / "history.csv"
    history_digest = hashlib.sha256(training_history_path.read_bytes()).hexdigest()
    best_epoch = last["best_epoch"]
    if not 1 <= best_epoch <= completed_epochs or checkpoint_epoch(best) != best_epoch:
        raise ValueError("best.pth 与 last.pth 的最佳轮次记录不一致")

    model = MLP(
        input_dim=784,
        hidden_dims=config["hidden_dims"],
        output_dim=10,
        activation=config["activation"],
        dropout=config["dropout"],
        batch_norm=config["batch_norm"],
    ).to(device)
    criterion = build_loss(config["loss"]).to(device)
    batch_size = config["batch_size"] if batch_size is None else batch_size
    if batch_size <= 0:
        raise ValueError("batch-size 必须大于 0")
    test_loader = get_test_loader(Path(__file__).resolve().parent / "data", batch_size)

    def score(checkpoint, expected_epoch):
        if checkpoint_epoch(checkpoint) != expected_epoch:
            raise ValueError(f"第 {expected_epoch} 轮检查点的轮次记录不一致")
        if normalized_config(checkpoint["config"]) != config:
            raise ValueError(f"第 {expected_epoch} 轮检查点的实验配置不一致")
        model.load_state_dict(checkpoint["model_state_dict"])
        # evaluate_model() 使用 eval()；测试损失不含 L2。
        loss, error = evaluate_model(model, test_loader, criterion, device)
        return {"epoch": expected_epoch, "test_loss": loss, "test_error": error}

    print(f"设备：{device}；测试集：{len(test_loader.dataset):,} 张")
    best_result = score(best, best_epoch)
    records = []
    missing_epochs = []
    for epoch in range(1, completed_epochs + 1):
        path = run_dir / "epochs" / f"epoch_{epoch:03d}.pth"
        if epoch == best_epoch:
            record = dict(best_result)
        elif path.exists():
            record = score(load_checkpoint(path), epoch)
        elif epoch == completed_epochs:
            record = score(last, epoch)
        else:
            missing_epochs.append(epoch)
            continue
        records.append(record)
        print(f"Epoch {epoch} | test_loss={record['test_loss']:.4f} | "
              f"test_error={record['test_error']:.2%}")

    if hashlib.sha256(training_history_path.read_bytes()).hexdigest() != history_digest:
        raise ValueError("评估期间训练记录发生了变化，请在训练结束后重新评估")

    # 全部评估成功后再替换输出，避免失败时留下半份 CSV。
    history_path = run_dir / "test_history.csv"
    history_temp = history_path.with_suffix(".tmp")
    with history_temp.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "test_loss", "test_error"])
        writer.writeheader()
        writer.writerows(records)
    history_temp.replace(history_path)

    summary = {
        "name": config["name"],
        "config": config,
        "num_params": sum(parameter.numel() for parameter in model.parameters()),
        "completed_epochs": completed_epochs,
        "best_epoch": best_epoch,
        "best_val_error": last["best_val_error"],
        "best_test_loss": best_result["test_loss"],
        "best_test_error": best_result["test_error"],
        "test_samples": len(test_loader.dataset),
        "evaluated_epochs": [record["epoch"] for record in records],
        "missing_epochs": missing_epochs,
        "complete_test_curve": not missing_epochs,
        "train_history_sha256": history_digest,
    }
    summary_path = run_dir / "test_summary.json"
    summary_temp = summary_path.with_suffix(".tmp")
    summary_temp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_temp.replace(summary_path)
    print(f"验证集选出的最佳模型：epoch {best_epoch} | "
          f"test_error={best_result['test_error']:.2%}")
    if missing_epochs:
        print(f"缺少第 {missing_epochs} 轮权重，已评估现存模型；缺失轮次不会补造数据。")
    print(f"已保存：{history_path}\n已保存：{summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="评估 Fashion-MNIST 实验的逐轮模型和最佳模型")
    parser.add_argument("--run", required=True, type=Path, help="实验目录，例如 results/baseline")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=None, help="默认沿用保存的批量大小")
    args = parser.parse_args()
    try:
        evaluate_run(args.run, args.device, args.batch_size)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
