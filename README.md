# 南开大学深度学习实验

本仓库目前提交实验一（Lab01）的代码、分析结果与 LaTeX 报告。根目录的个人 PyTorch 练习文件仅在本地保留，不纳入本次发布。

仓库地址：https://github.com/summerwind0131/nku_deep_learning

实验一的运行和结果复核说明见 [lab01/README.md](lab01/README.md)。

## 目录

```text
deep_learning/
├── lab01/
│   ├── config.py              # 命令行参数
│   ├── data.py                # Fashion-MNIST 加载与训练/验证划分
│   ├── model.py               # MLP、激活函数、Dropout、BN
│   ├── losses.py              # 交叉熵与概率 MSE
│   ├── engine.py              # 单轮训练与评估、显式 L2
│   ├── train.py               # 训练入口、日志、检查点及续训
│   ├── run_experiments.py     # 批量实验队列
│   ├── evaluate.py / plot.py  # 测试评估与单次实验绘图
│   ├── XOR.py                # XOR 附加实验
│   ├── analysis/             # 跨实验分析脚本与小型结果文件
│   ├── report/               # 报告源文件、证据与最终 PDF
│   ├── data/                 # 本地下载的数据集，不纳入 Git
│   └── results/              # 正式实验的少量 CSV/JSON 纳入 Git，权重等留在本地
└── requirements.txt
```

## 运行环境

原实验使用 Python 3.10.20、PyTorch 2.11.0 + CUDA 12.8。
`requirements.txt` 记录当前环境中的直接依赖版本；CPU/CUDA 构建需与运行机器匹配，它不是包含全部传递依赖的锁文件。

本机已有的 Conda 环境可以直接使用，在仓库根目录运行：

```bash
conda activate D:/conda_envs/dl-lab
cd lab01
```

在其他已经准备好 Python 和相应 PyTorch 构建的环境中，可安装其余依赖：

```bash
python -m pip install -r requirements.txt
```

## Lab01：多层感知机

以下命令在 `lab01/` 目录运行。首次使用会下载 Fashion-MNIST；训练集按固定种子分层划分为 50,000 张训练图像和 10,000 张验证图像，官方测试集独立保留。

```bash
# 查看全部参数
python train.py --help

# 开始一次新实验（名称应使用未存在的目录名）
python train.py --name baseline_new --epochs 20 --wandb-mode offline

# 从最近检查点恢复，目标总轮数为 40
python train.py --resume results/baseline_new/last.pth --epochs 40

# 确定实验配置并完成训练后，再评估测试集与绘图
python evaluate.py --run results/baseline_new
python plot.py --run results/baseline_new

# XOR
python XOR.py
```

`best.pth` 按最低验证误差选择；`last.pth` 用于继续训练。W&B 默认使用离线模式，无需登录。

批量实验可以先查看计划，再启动。默认包含 12 组对比，基础实验 20 轮、大模型实验 60 轮：

```bash
python run_experiments.py --prefix night02 --dry-run
python run_experiments.py --prefix night02 --evaluate-test
python analysis/analyze_experiments.py --prefix night02
```

重复使用相同 prefix 会续训或跳过已完成的训练；需要独立的新实验时应换一个 prefix。

## 修改实验报告

- 可直接编辑 `lab01/report/2412526-史峰源-实验1.tex`，其中已经嵌入图表，可独立用 XeLaTeX 编译。
- 当前交付 PDF 位于 [lab01/report/output/pdf/2412526-史峰源-实验1-提交版.pdf](lab01/report/output/pdf/2412526-史峰源-实验1-提交版.pdf)，首页包含仓库链接（按下方命令重新编译时，默认输出为不带“提交版”后缀的文件）。
- `report_template.tex` 和 `report_evidence.json` 是 `build_report.py` 的输入。
- **运行 `build_report.py` 会覆盖最终 `.tex` 文件。** 直接修改最终源文件后，应将需要长期保留的正文修改同步到模板，再考虑重新生成。
- 修改模板不会自动更新最终 `.tex` 或 PDF。`prepare_report_data.py` 还依赖本地数据集与训练结果；它与单独编译最终 `.tex` 是不同步骤。

在 `lab01/report/` 目录编译已编辑的最终源文件：

```bash
xelatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf "2412526-史峰源-实验1.tex"
xelatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf "2412526-史峰源-实验1.tex"
```

使用两次编译更新交叉引用。编译后的 PDF 应再检查分页和图表位置。

## Git 跟踪范围

保留 Lab01 的 Python 代码、报告源文件、提交版 PDF、图表及 CSV/JSON 分析摘要。`results/night01_*/` 仅保留 `config.json`、`history.csv`、`test_history.csv`、`test_summary.json`，另保留正式实验计划。数据集、模型权重、其他训练输出、W&B 日志、Python 缓存、LaTeX 中间文件、报告历史备份和临时预览由 `.gitignore` 排除。

忽略规则不会删除本地文件；克隆仓库也不会获得训练权重，续训前需另行准备对应的检查点。`.gitattributes` 统一源代码换行，但保留正式结果文件的原始字节，确保 CSV 的 SHA256 校验跨平台有效。

老师提交用的源码压缩包另行保存在本地 `lab01/submission/`，包含实验代码、运行说明及上述小型结果。该目录不重复上传到 GitHub；提交作业时使用压缩包和提交版 PDF。
