import torch

from config import parse_args
from model import MLP
from data import get_data_loader
from engine import train_one_epoch,evaluate
from losses import build_loss

import csv
from pathlib import Path
from uuid import uuid4
import json
import random
import warnings

import numpy as np
import wandb




def capture_rng_state(train_loader):
    """保存随机状态；NumPy 的数组转换为列表以兼容 weights_only=True。"""
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (
            numpy_state[0],
            numpy_state[1].tolist(),
            int(numpy_state[2]),
            int(numpy_state[3]),
            float(numpy_state[4]),
        ),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "train_generator": train_loader.generator.get_state(),
    }


def restore_rng_state(state, train_loader):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((
        numpy_state[0],
        np.asarray(numpy_state[1], dtype=np.uint32),
        numpy_state[2],
        numpy_state[3],
        numpy_state[4],
    ))
    # map_location=device 可能将这些状态张量加载到 GPU；恢复时需放回 CPU。
    torch.set_rng_state(state["torch"].cpu())
    train_loader.generator.set_state(state["train_generator"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            warnings.warn("GPU 数量与检查点不一致，无法精确恢复 CUDA 随机状态")
        else:
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def save_history(history, path):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_total_loss",
                "train_error",
                "val_loss",
                "val_error"
            ],
        )
        writer.writeheader()
        writer.writerows(history)


def save_epoch_model(model, config, epoch, epoch_dir):
    """逐轮评估只需要模型配置和权重，不重复保存优化器、历史和随机状态。"""
    path = epoch_dir / f"epoch_{epoch:03d}.pth"
    temp_path = path.with_suffix(".tmp")
    torch.save({
        "epoch": epoch,
        "config": config,
        "model_state_dict": model.state_dict(),
    }, temp_path)
    temp_path.replace(path)


def main():
    args=parse_args()

    base_dir=Path(__file__).resolve().parent
    device=torch.device(
        "cuda" if torch.cuda.is_available()else "cpu"
    )
    checkpoint=None

    if args.resume:
        checkpoint_path = Path(args.resume).resolve()
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )

        # 续训时以保存的配置为准
        config = checkpoint["config"]
        if config.get("data_protocol") != "fashion_stratified_50k_10k_v1":
            raise ValueError("该检查点未使用当前验证集划分，请新建实验从头训练")
        run_dir = checkpoint_path.parent
        start_epoch = checkpoint["completed_epochs"]
        history = checkpoint["history"]
        run_id = checkpoint["wandb_run_id"]

        if "best_val_error" not in checkpoint or "best_epoch" not in checkpoint:
            raise ValueError("旧检查点没有最佳模型记录，请用新实验名从头训练")

        best_val_error = checkpoint["best_val_error"]
        best_epoch = checkpoint["best_epoch"]

        if args.epochs <= start_epoch:
            raise ValueError(
                f"已经完成 {start_epoch} 轮，"
                f"--epochs 必须大于 {start_epoch}"
            )

        print("恢复原实验配置；本次只使用 --epochs 设置目标总轮数")

    else:
        # 实验名只作为目录名使用
        if (
            not args.name
            or args.name in {".", ".."}
            or any(c in args.name for c in '<>:"/\\|?*')
        ):
            raise ValueError("请使用简单的实验名，例如 baseline 或 small")

        config = {
            "name": args.name,
            "project": args.project,
            "hidden_dims": args.hidden_dims,
            "activation": args.activation,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "split_seed": args.split_seed,
            "optimizer": "Adam",
            "loss": args.loss,
            "l2": args.l2,
            "dropout": args.dropout,
            "batch_norm": args.batch_norm,
            "data_protocol":"fashion_stratified_50k_10k_v1"
        }

        run_dir = base_dir / "results" / args.name

        if run_dir.exists():
            raise FileExistsError(
                f"{run_dir} 已存在，请换一个 --name，"
                "或者使用 --resume 继续训练"
            )

        run_dir.mkdir(parents=True)
        start_epoch = 0
        history = []
        run_id = uuid4().hex
        best_val_error = float("inf")
        best_epoch = 0

    config.setdefault("l2", 0.0)
    config.setdefault("dropout", 0.0)
    config.setdefault("batch_norm", False)

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])

    model=MLP(
        input_dim=784,
        hidden_dims=config["hidden_dims"],
        output_dim=10,
        activation=config["activation"],
        dropout=config["dropout"],
        batch_norm=config["batch_norm"]
    ).to(device)
    optimizer=torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=0.0,
    )
    criterion=build_loss(config["loss"]).to(device)
    train_loader,val_loader,test_loader=get_data_loader(
        data_dir=base_dir/"data",
        batch_size=config["batch_size"],
        split_seed=config["split_seed"],
        train_seed=config["seed"]
    )

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch_dir = run_dir / "epochs"
    epoch_dir.mkdir(exist_ok=True)
    # 旧实验续训时，至少补存当前恢复轮次；更早的权重无法从 last.pth 还原。
    if checkpoint is not None and not (epoch_dir / f"epoch_{start_epoch:03d}.pth").exists():
        save_epoch_model(model, config, start_epoch, epoch_dir)

    rng_state = checkpoint.get("rng_state") if checkpoint is not None else None
    if rng_state is None:
        if checkpoint is not None:
            warnings.warn(
                "旧检查点没有随机状态，仍可续训，但本次续训无法与连续训练精确对齐"
            )
        rng_state = capture_rng_state(train_loader)
    num_params=sum(p.numel()for p in model.parameters())
    print(f"设备：{device}，参数量：{num_params:,}")

    with (run_dir / "config.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


    with wandb.init(
        project=config["project"],
        name=config["name"],
        id=run_id,
        mode=args.wandb_mode,
        resume=(
            "must"
            if checkpoint is not None and args.wandb_mode == "online"
            else None
        ),
        config=config,
        dir=str(run_dir),
    ) as run:
        # 指定曲线使用 epoch 作为横轴
        run.define_metric("epoch")
        for metric in [
            "train_loss",
            "train_total_loss",
            "train_error",
            "val_loss",
            "val_error",
        ]:
            run.define_metric(metric, step_metric="epoch")

        run.summary["num_params"] = num_params
        run.summary["target_epochs"] = args.epochs

        # 模型、数据和 W&B 都初始化后再恢复，避免初始化消耗已恢复的随机状态。
        restore_rng_state(rng_state, train_loader)

        for epoch in range(start_epoch, args.epochs):
            train_loss, train_total_loss = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                l2=config["l2"],
            )

            _, train_error = evaluate(
                model, train_loader, criterion, device
            )
            val_loss, val_error = evaluate(
                model, val_loader, criterion, device
            )


            record = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_total_loss": train_total_loss,
                "train_error": train_error,
                "val_loss":val_loss,
                "val_error": val_error,
            }
            history.append(record)

            # 验证误差相同时，保留较早的模型
            is_best = val_error < best_val_error
            if is_best:
                best_val_error = val_error
                best_epoch = epoch + 1

            run.summary["best_val_error"] = best_val_error
            run.summary["best_epoch"] = best_epoch
            save_history(history, run_dir / "history.csv")
            run.log(record)

            checkpoint_data = {
                "config": config,
                "completed_epochs": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "wandb_run_id": run.id,
                "best_val_error": best_val_error,
                "best_epoch": best_epoch,
                "rng_state": capture_rng_state(train_loader),
            }

            save_epoch_model(model, config, epoch + 1, epoch_dir)

            if is_best:
                best_temp = run_dir / "best.tmp"
                torch.save(checkpoint_data, best_temp)
                best_temp.replace(run_dir / "best.pth")

            last_temp = run_dir / "last.tmp"
            torch.save(checkpoint_data, last_temp)
            last_temp.replace(run_dir / "last.pth")

            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"data_loss={train_loss:.4f} | "
                f"total_loss={train_total_loss:.4f} | "
                f"train_error={train_error:.2%} | "
                f"val_error={val_error:.2%}"
            )



if __name__=="__main__":
    main()
