import argparse

def parse_args():
    parser=argparse.ArgumentParser(description="Fashion-MNIST实验")

    parser.add_argument("--name",type=str,default="baseline")
    parser.add_argument("--epochs",type=int,default=20)
    parser.add_argument("--batch-size",type=int,default=64)
    parser.add_argument("--lr", "--learning-rate", dest="lr", type=float, default=0.001)

    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[256,128],
    )

    parser.add_argument(
        "--activation",
        choices=["relu","tanh","sigmoid"],
        default="relu",
    )
    parser.add_argument(
        "--loss",
        choices=["cross_entropy","mse"],
        default="cross_entropy",
    )
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--split-seed",type=int,default=42)
    parser.add_argument("--project",type=str,default="fashion-mnist-lab01")
    parser.add_argument("--resume",type=str,default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="offline",
    )
    parser.add_argument(
        "--l2",
        type=float,
        default=0.0,
        help="L2 正则化系数，0 表示不使用"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="隐藏层 Dropout 概率，0 表示关闭",

    )
    parser.add_argument(
        "--batch-norm",
        action="store_true",
        help="在隐藏层启用 BatchNorm",
    )
    

    args=parser.parse_args()
    
    if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0:
        parser.error("epochs、batch-size 和 lr 必须大于 0")

    if len(args.hidden_dims) < 2 or any(n <= 0 for n in args.hidden_dims):
        parser.error("至少设置两个隐藏层，每层神经元数必须大于 0")

    if args.l2<0:
        parser.error("l2必须大于等于 0")

    if not 0.0 <= args.dropout < 1.0:
        parser.error("dropout 必须满足 0 <= dropout < 1")

    if args.batch_norm and args.batch_size < 2:
        parser.error("启用 BatchNorm 时，batch-size 必须至少为 2")

    return args
