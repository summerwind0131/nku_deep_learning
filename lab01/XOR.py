import torch
from torch import nn
from torch.nn import functional as F
torch.manual_seed(1)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lc1=nn.Linear(2,5)
        self.lc2=nn.Linear(5,1)

    def forward(self,x):
        x=self.lc1(x)
        x=F.relu(x)
        x=self.lc2(x)
        return x

x = torch.tensor([
    [0., 0.],
    [0., 1.],
    [1., 0.],
    [1., 1.]
])

y = torch.tensor([
    [0.],
    [1.],
    [1.],
    [0.]
])

model=Model()
criterion=nn.MSELoss()
optimizer=torch.optim.SGD(
    model.parameters(),
    lr=0.1
)

for epoch in range(5000):
    pred=model(x)

    loss=criterion(pred,y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if epoch % 500 == 0:
        print(
            f"epoch={epoch}, "
            f"loss={loss.item():.6f}"
        )

with torch.no_grad():
    pred = model(x)

    print("\n预测值：")
    print(pred)

    print("\n二值化结果：")
    print((pred > 0.5).int())