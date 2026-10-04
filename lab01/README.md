# 实验一：多层感知机

课程：深度学习实验；姓名：史峰源；学号：2412526；指导教师：李欢。

代码仓库：https://github.com/summerwind0131/nku_deep_learning

本实验实现 Fashion-MNIST 分类，并对比网络结构、激活函数、损失函数、L2、Dropout 和 Batch Normalization；`XOR.py` 为 XOR 附加题。

## 文件说明

| 文件或目录 | 用途 |
| --- | --- |
| `config.py`、`data.py` | 参数解析、数据加载与固定训练/验证划分 |
| `model.py`、`losses.py` | MLP 结构、激活、正则化模块及分类损失 |
| `engine.py`、`train.py` | 训练、评估、指标记录、模型保存与续训 |
| `run_experiments.py` | 12 组实验的参数与执行队列 |
| `evaluate.py`、`plot.py` | 已保存模型的测试评估、单组曲线绘制 |
| `XOR.py` | XOR 网络与训练 |
| `analysis/analyze_experiments.py` | 从 CSV 和 JSON 结果重新生成对比表与图 |
| `results/night01_*/` | 12 组正式实验的配置、逐轮指标和测试摘要 |
| `results/_batch/night01/plan.json` | 正式实验计划 |
| `analysis/night01/` | 汇总表与对比图 |

源码压缩包保留上述实验代码和小型结果。完整数据集、模型权重、W&B 日志及临时文件不在提交包内；实验报告 PDF 单独提交，报告源文件可在 GitHub 的 `lab01/report/` 中查看。

## 环境准备

在解压后的 `lab01/` 目录运行以下命令。原实验环境为 Python 3.10.20，依赖版本见 `requirements.txt`：

```bash
python -m pip install -r requirements.txt
```

原实验使用 CUDA 12.8 版本的 PyTorch；安装时可按本机条件选择 CPU 或 CUDA 构建。W&B 默认采用离线模式，不需要登录。

`train.py` 可按机器条件使用 CPU 或 CUDA；当前批量队列 `run_experiments.py` 的实际训练要求 CUDA，CPU 环境可分别调用 `train.py`。查看队列计划（`--dry-run`）及复核已有 CSV 不要求 CUDA。

## 直接核对已有结果（无需权重和数据集）

```bash
python analysis/analyze_experiments.py --prefix night01
python plot.py --run results/night01_baseline
```

分析脚本检查 12 组实验的轮次是否完整、训练指标哈希及最佳轮次是否与摘要一致，再生成汇总 CSV 和图。已有结果共 440 个 epoch，基础组训练 20 轮，大模型组训练 60 轮。

`history.csv` 包含训练损失、总目标、训练误差、验证损失和验证误差；`test_history.csv` 包含对应轮次的测试损失与测试误差。误差在 CSV 中以 0–1 小数保存，汇总表中以百分数表示。“最佳轮”由最低验证误差选出，不按测试误差选择。

## 重新运行实验

```bash
# 查看参数
python train.py --help

# 新实验示例；首次训练自动下载 Fashion-MNIST
python train.py --name baseline_reproduce --epochs 20 --wandb-mode offline

# 所有 12 组实验：先查看计划，再执行
python run_experiments.py --prefix reproduce --dry-run
python run_experiments.py --prefix reproduce --evaluate-test

# 重新汇总
python analysis/analyze_experiments.py --prefix reproduce

# XOR 附加题
python XOR.py
```

50,000 张训练图像与 10,000 张验证图像由官方训练集分层划分，划分种子为 42；官方测试集单独保留。模型使用 Adam、学习率 0.001、批量大小 64。各组具体差异可查看 `run_experiments.py` 或 `plan.json`。

**重新训练请使用新的实验名称或 prefix。** 随包提供的 `night01` 目录只有结果，没有 `last.pth`，不能直接续训。自行训练后可以执行：

```bash
python train.py --resume results/baseline_reproduce/last.pth --epochs 40
python evaluate.py --run results/baseline_reproduce
python plot.py --run results/baseline_reproduce
```

其中 `--epochs 40` 表示训练到总共 40 轮。`last.pth` 用于续训；`best.pth` 保存验证误差最低的模型。
