from torch import nn
from torch.nn import functional as F

class ClassificationMSELoss(nn.Module):
    def forward(self,logtis,labels):
        probabilities=F.softmax(logtis,dim=1)
        targets=F.one_hot(
            labels,
            num_classes=logtis.shape[1]
        ).to(dtype=logtis.dtype)

        return F.mse_loss(probabilities,targets)


def build_loss(name):
    if name=="cross_entropy":
        return nn.CrossEntropyLoss()
    if name=="mse":
        return ClassificationMSELoss()

    raise ValueError(f"不支持的损失函数：{name}")



