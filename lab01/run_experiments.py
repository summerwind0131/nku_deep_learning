"""离线串行实验队列；重复运行同一 prefix 会续训或跳过已完成训练。"""

import argparse
from contextlib import contextmanager
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def make_jobs(epochs=20, extra_epochs=60):
    # 基础对照只改变一个因素；附加题在相同的大模型上比较。
    baseline = {
        "hidden_dims": [256, 128], "activation": "relu", "loss": "cross_entropy",
        "l2": 0.0, "dropout": 0.0, "batch_norm": False,
    }
    specifications = [
        ("baseline", epochs, {}),
        ("narrow", epochs, {"hidden_dims": [64, 32]}),
        ("wide", epochs, {"hidden_dims": [512, 256]}),
        ("deep", epochs, {"hidden_dims": [256, 128, 64]}),
        ("tanh", epochs, {"activation": "tanh"}),
        ("sigmoid", epochs, {"activation": "sigmoid"}),
        ("mse", epochs, {"loss": "mse"}),
        ("large_plain", extra_epochs, {}),
        ("large_l2_1e-4", extra_epochs, {"l2": 0.0001}),
        ("large_l2_1e-3", extra_epochs, {"l2": 0.001}),
        ("large_dropout", extra_epochs, {"dropout": 0.2}),
        ("large_bn", extra_epochs, {"batch_norm": True}),
    ]
    jobs = []
    for label, target_epochs, changes in specifications:
        config = {**baseline, **changes}
        if label.startswith("large_"):
            config["hidden_dims"] = [1024, 1024, 512]
        jobs.append({"label": label, "epochs": target_epochs, "config": config})
    return jobs


def training_command(base_dir, name, job, resume=None):
    command = [sys.executable, "-u", str(base_dir / "train.py"),
               "--epochs", str(job["epochs"]), "--wandb-mode", "offline"]
    if resume is not None:
        return command + ["--resume", str(resume)]
    config = job["config"]
    command += ["--name", name, "--hidden-dims", *map(str, config["hidden_dims"]),
                "--activation", config["activation"], "--loss", config["loss"],
                "--l2", str(config["l2"]), "--dropout", str(config["dropout"]),
                "--seed", "42", "--split-seed", "42", "--batch-size", "64", "--lr", "0.001"]
    if config["batch_norm"]:
        command.append("--batch-norm")
    return command


def resume_epoch(run_dir, job):
    if not run_dir.exists():
        return 0
    checkpoint_path = run_dir / "last.pth"
    if not checkpoint_path.exists():
        raise ValueError(f"{run_dir} 已存在但没有 last.pth；请检查日志，或换 --prefix")
    import torch
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    expected = {**job["config"], "seed": 42, "split_seed": 42, "batch_size": 64,
                "lr": 0.001, "optimizer": "Adam", "data_protocol": "fashion_stratified_50k_10k_v1"}
    for key, value in expected.items():
        if checkpoint["config"].get(key) != value:
            raise ValueError(f"{run_dir.name} 的 {key} 与本次计划不一致，请换 --prefix")
    return checkpoint["completed_epochs"]


@contextmanager
def keep_awake():
    # 请求期间防止 Windows 空闲睡眠；显示器仍可自动关闭，不修改电源方案。
    function = None
    if os.name == "nt":
        function = ctypes.WinDLL("kernel32", use_last_error=True).SetThreadExecutionState
        function.argtypes = [ctypes.c_uint]
        function.restype = ctypes.c_uint
        if not function(0x80000000 | 0x00000001):
            raise OSError(ctypes.get_last_error(), "无法请求防止空闲睡眠")
    try:
        yield
    finally:
        if function is not None:
            function(0x80000000)


def run_command(command, cwd, log_path):
    environment = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n" + json.dumps(command, ensure_ascii=False) + "\n")
        log.flush()
        process = subprocess.Popen(command, cwd=cwd, env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace")
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
        finally:
            process.stdout.close()


def save_status(path, status):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description="Fashion-MNIST 离线实验队列")
    parser.add_argument("--prefix", default="night01", help="实验名前缀；相同前缀用于恢复队列")
    parser.add_argument("--epochs", type=int, default=20, help="基础对照训练轮数")
    parser.add_argument("--extra-epochs", type=int, default=60, help="大模型及正则化对照训练轮数")
    parser.add_argument("--dry-run", action="store_true", help="只显示实验计划，不执行")
    parser.add_argument("--evaluate-test", action="store_true", help="方案确定后才启用：训练完统一评估测试集")
    args = parser.parse_args()
    if args.epochs <= 0 or args.extra_epochs <= 0:
        parser.error("训练轮数必须大于 0")
    if not args.prefix or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in args.prefix):
        parser.error("prefix 请仅使用英文字母、数字、下划线或连字符")

    base_dir = Path(__file__).resolve().parent
    jobs = make_jobs(args.epochs, args.extra_epochs)
    print(f"共 {len(jobs)} 组，计划总轮数 {sum(job['epochs'] for job in jobs)}；固定 seed=42、split_seed=42")
    for job in jobs:
        print(f"{args.prefix}_{job['label']}: epochs={job['epochs']}, {job['config']}")
    if args.dry_run:
        return 0

    import torch
    if not torch.cuda.is_available():
        parser.error("当前 Python 环境不能使用 CUDA，请先激活 dl-lab")
    print(f"GPU：{torch.cuda.get_device_name(0)}；按顺序运行，每次仅启动一个训练进程")
    batch_dir = base_dir / "results" / "_batch" / args.prefix
    batch_dir.mkdir(parents=True, exist_ok=True)
    (batch_dir / "plan.json").write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")
    status_path = batch_dir / "status.json"
    status = {}
    with keep_awake():
        try:
            for index, job in enumerate(jobs, 1):
                name = f"{args.prefix}_{job['label']}"
                run_dir = base_dir / "results" / name
                log_path = batch_dir / f"{name}.log"
                started = time.perf_counter()
                entry = {"status": "running", "phase": "train", "log": str(log_path)}
                status[name] = entry
                save_status(status_path, status)
                print(f"\n[{index}/{len(jobs)}] {name}", flush=True)
                try:
                    completed = resume_epoch(run_dir, job)
                    if completed > job["epochs"]:
                        raise ValueError(f"已训练 {completed} 轮，超过本次 {job['epochs']} 轮预算，请换 --prefix")
                    if completed < job["epochs"]:
                        command = training_command(base_dir, name, job,
                            run_dir / "last.pth" if completed else None)
                        code = run_command(command, base_dir, log_path)
                        if code:
                            raise RuntimeError(f"训练退出码 {code}")
                    else:
                        print(f"已完成 {completed} 轮，跳过训练")
                    entry["status"] = "trained"
                except Exception as exc:
                    entry.update(status="failed", error=str(exc))
                    with log_path.open("a", encoding="utf-8") as log:
                        log.write(f"\n{type(exc).__name__}: {exc}\n")
                    print(f"失败：{exc}；继续下一组。详情见 {log_path}")
                entry["seconds"] = round(time.perf_counter() - started, 1)
                save_status(status_path, status)

            # 先完成全部训练，随后统一导出，避免测试结果影响训练过程。
            for name, entry in status.items():
                if entry["status"] != "trained":
                    continue
                run_dir = base_dir / "results" / name
                scripts = ["evaluate.py", "plot.py"] if args.evaluate_test else ["plot.py"]
                try:
                    for script in scripts:
                        entry["phase"] = script
                        save_status(status_path, status)
                        code = run_command([sys.executable, "-u", str(base_dir / script),
                                            "--run", str(run_dir)], base_dir, Path(entry["log"]))
                        if code:
                            raise RuntimeError(f"{script} 退出码 {code}")
                    entry.update(status="complete", phase="done")
                except Exception as exc:
                    entry.update(status="failed", error=str(exc))
                    print(f"导出失败：{name}: {exc}；继续下一组。")
                save_status(status_path, status)
        except KeyboardInterrupt:
            for entry in status.values():
                if entry["status"] in {"running", "trained"}:
                    entry["status"] = "interrupted"
            save_status(status_path, status)
            print("\n队列已停止；再次运行相同命令可从已有 last.pth 恢复。")
            return 130
    failures = [name for name, entry in status.items() if entry["status"] != "complete"]
    print(f"\n队列结束：成功 {len(status) - len(failures)}/{len(status)}；状态：{status_path}")
    if failures:
        print(f"失败实验：{failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
