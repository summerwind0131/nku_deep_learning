"""Read completed experiments and export comparison tables and research figures.

Run from any directory with the same Python environment used for training.
This script only writes derived files under lab01/analysis/<prefix>.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "baseline": "Baseline", "narrow": "Narrow", "wide": "Wide",
    "deep": "Deep", "tanh": "Tanh", "sigmoid": "Sigmoid", "mse": "MSE",
    "large_plain": "None", "large_l2_1e-4": "L2 1e-4",
    "large_l2_1e-3": "L2 1e-3", "large_dropout": "Dropout 0.2", "large_bn": "BN",
}
COLORS = ["#3368A5", "#B77D25", "#C46C43", "#698440", "#A55D82"]
STYLES = ["-", "--", "-.", ":", "-"]


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as file:
        return [{key: int(value) if key == "epoch" else float(value)
                 for key, value in row.items()} for row in csv.DictReader(file)]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected(run, budget=None):
    rows = run["history"] if budget is None else run["history"][:budget]
    best = min(rows, key=lambda row: row["val_error"])
    test = run["tests"][best["epoch"]]
    return {"epoch": best["epoch"], "train_error": best["train_error"],
            "val_error": best["val_error"], "test_error": test["test_error"]}


def bars(ax, runs, labels, budget, title):
    points = [selected(run, budget) for run in runs]
    x = list(range(len(points)))
    width = 0.34
    for offset, field, label, color, hatch in (
        (-width / 2, "val_error", "Validation", COLORS[0], ""),
        (width / 2, "test_error", "Test", COLORS[1], "//"),
    ):
        values = [point[field] * 100 for point in points]
        rects = ax.bar([pos + offset for pos in x], values, width,
                       color=color, label=label, hatch=hatch, edgecolor="white")
        ax.bar_label(rects, labels=[f"{value:.2f}" for value in values],
                     padding=3, fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 14)
    ax.set_ylabel("Classification error (%)")
    ax.set_title(title, loc="left", fontweight="bold", pad=12)
    ax.legend(frameon=False, loc="upper right", ncols=2, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)


def error_lines(ax, runs, labels, title):
    for index, (run, label) in enumerate(zip(runs, labels)):
        rows = run["history"][:20]
        ax.plot([row["epoch"] for row in rows], [row["val_error"] for row in rows],
                color=COLORS[index], linestyle=STYLES[index], label=label,
                linewidth=1.7)
    ax.set_ylim(0.08, 0.165)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.set_xlim(1, 20)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_ylabel("Validation classification error")
    ax.set_xlabel("Epoch")
    ax.set_title(title, loc="left", fontweight="bold", pad=12)
    ax.legend(frameon=False, ncols=2, fontsize=9)
    ax.grid(alpha=0.25)


def save_figure(fig, target):
    fig.savefig(target.with_suffix(".png"), dpi=200)
    fig.savefig(target.with_suffix(".svg"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="night01")
    args = parser.parse_args()
    if not args.prefix or Path(args.prefix).name != args.prefix or any(
        character in args.prefix for character in "/\\:"
    ):
        parser.error("prefix must be a single directory name")
    plan_path = ROOT / "results" / "_batch" / args.prefix / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    runs = {}
    provenance = [{"path": str(plan_path), "sha256": digest(plan_path)}]
    for job in plan:
        directory = ROOT / "results" / f"{args.prefix}_{job['label']}"
        summary_path = directory / "test_summary.json"
        history_path = directory / "history.csv"
        test_path = directory / "test_history.csv"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        history = read_csv(history_path)
        tests = read_csv(test_path)
        expected = list(range(1, job["epochs"] + 1))
        if [row["epoch"] for row in history] != expected or [row["epoch"] for row in tests] != expected:
            raise ValueError(f"Incomplete epoch coverage: {directory}")
        if not summary["complete_test_curve"] or summary["missing_epochs"]:
            raise ValueError(f"Incomplete test curve: {directory}")
        if summary["train_history_sha256"] != digest(history_path):
            raise ValueError(f"Stale test results: {directory}")
        if summary["test_samples"] != 10000:
            raise ValueError(f"Unexpected test population: {directory}")
        config = summary["config"]
        for key, value in job["config"].items():
            if config[key] != value:
                raise ValueError(f"Configuration mismatch for {key}: {directory}")
        for key, value in {"seed": 42, "split_seed": 42, "batch_size": 64,
                           "lr": 0.001, "optimizer": "Adam",
                           "data_protocol": "fashion_stratified_50k_10k_v1"}.items():
            if config[key] != value:
                raise ValueError(f"Unexpected shared configuration for {key}: {directory}")
        run = {"label": job["label"], "summary": summary, "history": history,
               "tests": {row["epoch"]: row for row in tests}}
        chosen = selected(run)
        if chosen["epoch"] != summary["best_epoch"] or chosen["val_error"] != summary["best_val_error"]:
            raise ValueError(f"Best checkpoint selection mismatch: {directory}")
        if chosen["test_error"] != summary["best_test_error"]:
            raise ValueError(f"Test summary mismatch: {directory}")
        runs[job["label"]] = run
        provenance.extend({"path": str(path), "sha256": digest(path)}
                          for path in (summary_path, history_path, test_path))

    output = ROOT / "analysis" / args.prefix
    output.mkdir(parents=True, exist_ok=True)
    table = []
    for run in runs.values():
        summary = run["summary"]
        config = summary["config"]
        best = selected(run)
        first20 = selected(run, 20)
        final = run["history"][-1]
        minimum = min(run["history"], key=lambda row: row["val_loss"])
        table.append({
            "run": summary["name"], "hidden_dims": "-".join(map(str, config["hidden_dims"])),
            "activation": config["activation"], "loss": config["loss"],
            "l2": config["l2"], "dropout": config["dropout"], "batch_norm": config["batch_norm"],
            "num_params": summary["num_params"], "epochs": summary["completed_epochs"],
            "best_epoch": best["epoch"], "best_train_error_pct": best["train_error"] * 100,
            "best_val_error_pct": best["val_error"] * 100, "best_test_error_pct": best["test_error"] * 100,
            "first20_best_epoch": first20["epoch"], "first20_best_val_error_pct": first20["val_error"] * 100,
            "first20_best_test_error_pct": first20["test_error"] * 100,
            "final_train_error_pct": final["train_error"] * 100,
            "final_val_error_pct": final["val_error"] * 100,
            "final_test_error_pct": run["tests"][final["epoch"]]["test_error"] * 100,
            "final_val_train_gap_pp": (final["val_error"] - final["train_error"]) * 100,
            "min_val_loss_epoch": minimum["epoch"], "min_val_loss": minimum["val_loss"],
            "final_val_loss": final["val_loss"],
        })
    with (output / "comparison_results.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    evidence = {
        "prefix": args.prefix, "train_samples": 50000, "validation_samples": 10000,
        "test_samples": 10000, "seed": 42, "split_seed": 42,
        "selection": "Earliest minimum validation classification error within each indicated epoch budget; test data never selects checkpoints.",
        "structure_budget": 20, "activation_budget": 20, "loss_budget": 20, "regularization_budget": 60,
        "limitations": ["One random seed; no run-to-run uncertainty estimates.",
                        "Depth and parameter count are not independently isolated.",
                        "Error metrics use model.eval(); training losses are accumulated during parameter updates.",
                        "CE and MSE numerical loss scales cannot be compared directly.",
                        "L2 is explicit 0.5 * coefficient * sum of squared Linear weights, excluding biases and BN parameters."],
        "results": table, "sources": provenance,
    }
    (output / "comparison_evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    structures = [runs[key] for key in ("narrow", "baseline", "wide", "deep", "large_plain")]
    structure_labels = ["64-32", "256-128", "512-256", "256-128-64", "1024-1024-512"]
    activations = [runs[key] for key in ("baseline", "tanh", "sigmoid")]
    losses = [runs[key] for key in ("baseline", "mse")]
    regularization = [runs[key] for key in ("large_plain", "large_l2_1e-4", "large_l2_1e-3", "large_dropout", "large_bn")]
    regularization_labels = [LABELS[run["label"]] for run in regularization]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    bars(axes[0, 0], structures, structure_labels, 20, "A. Hidden-layer structure | first 20 epochs")
    bars(axes[0, 1], activations, ["ReLU", "Tanh", "Sigmoid"], 20, "B. Activation | 20 epochs")
    bars(axes[1, 0], losses, ["Cross entropy", "Softmax + one-hot MSE"], 20, "C. Loss function | 20 epochs")
    bars(axes[1, 1], regularization, regularization_labels, 60, "D. Large-model variants | 60 epochs")
    axes[0, 0].tick_params(axis="x", labelsize=9)
    axes[1, 1].tick_params(axis="x", labelsize=9)
    fig.suptitle("Fashion-MNIST: validation-selected checkpoint comparisons", fontsize=16, y=0.98)
    fig.text(0.5, 0.025, "Each panel selects the earliest minimum validation error within its indicated budget.\n"
             "Shared split: 50k train / 10k validation / 10k test; Adam, lr=0.001, batch=64; one seed (42).", ha="center", fontsize=10)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.11, top=0.91, hspace=0.40, wspace=0.25)
    save_figure(fig, output / "comparison_errors")

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    error_lines(axes[0, 0], structures, structure_labels, "A. Structure | validation error, first 20 epochs")
    error_lines(axes[0, 1], activations, ["ReLU", "Tanh", "Sigmoid"], "B. Activation | validation error")
    error_lines(axes[1, 0], losses, ["Cross entropy", "Softmax + one-hot MSE"], "C. Loss function | validation error")
    ax = axes[1, 1]
    for index, (run, label) in enumerate(zip(regularization, regularization_labels)):
        rows = run["history"]
        ax.plot([row["epoch"] for row in rows], [row["val_loss"] for row in rows],
                color=COLORS[index], linestyle=STYLES[index], linewidth=1.7, label=label)
    ax.set_xlim(1, 60)
    ax.set_ylim(0, 0.85)
    ax.set_xticks([1, 10, 20, 30, 40, 50, 60])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation cross-entropy loss (no L2 term)")
    ax.set_title("D. Large-model variants | validation data loss", loc="left", fontweight="bold", pad=12)
    ax.legend(frameon=False, ncols=2, fontsize=9)
    ax.grid(alpha=0.25)
    fig.suptitle("Fashion-MNIST: validation learning curves", fontsize=16, y=0.98)
    fig.text(0.5, 0.025, "Panels A-C share the same error scale; panel D shows CE loss for all five 60-epoch variants.\n"
             "Loss selection and classification-error selection are different objectives; one seed (42).", ha="center", fontsize=10)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.11, top=0.91, hspace=0.38, wspace=0.28)
    save_figure(fig, output / "comparison_curves")
    print(f"Saved comparisons to {output}")
    for row in table:
        print(f"{row['run']}: params={row['num_params']:,}, best={row['best_epoch']}, "
              f"val={row['best_val_error_pct']:.2f}%, test={row['best_test_error_pct']:.2f}%, "
              f"final gap={row['final_val_train_gap_pp']:.2f} pp")


if __name__ == "__main__":
    main()
