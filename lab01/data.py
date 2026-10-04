from torchvision import datasets,transforms
from torch.utils.data import DataLoader,Subset
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
def get_data_loader(data_dir,batch_size,
                    split_seed=42,
                    train_seed=42):
    transform=transforms.ToTensor()
    full_dataset=datasets.FashionMNIST(
        root=data_dir,
        train=True,
        transform=transform,
        download=True
    )
    train_indices,val_indices=train_test_split(
        list(range(len(full_dataset))),
        test_size=10000,
        random_state=split_seed,
        stratify=full_dataset.targets.numpy()
    )
    train_dataset=Subset(full_dataset,train_indices)
    val_dataset=Subset(full_dataset,val_indices)


    train_generator=torch.Generator().manual_seed(train_seed)
    train_loader=DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator
    )
    val_loader=DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = get_test_loader(data_dir, batch_size)

    return train_loader,val_loader,test_loader


def get_test_loader(data_dir, batch_size):
    """独立评估时只加载官方测试集。"""
    test_dataset=datasets.FashionMNIST(
        root=data_dir,
        train=False,
        transform=transforms.ToTensor(),
        download=True
    )
    test_loader=DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return test_loader


if __name__=="__main__":
    data_dir = Path(__file__).resolve().parent / "data"
    train_loader,val_loader,test_loader=get_data_loader(
        data_dir=data_dir,
        batch_size=64
    )
    print("训练集：", len(train_loader.dataset))
    print("验证集：", len(val_loader.dataset))
    print("测试集：", len(test_loader.dataset))
