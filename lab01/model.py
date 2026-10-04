import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self,input_dim,hidden_dims,output_dim,
                 activation="relu",
                 dropout=0.0,
                 batch_norm=False):
        super().__init__()

        # 激活函数
        activation_classes={
            "relu":nn.ReLU,
            "tanh":nn.Tanh,
            "sigmoid":nn.Sigmoid,
        }
        activation_class=activation_classes[activation]


        # 层数
        layers=[nn.Flatten(start_dim=1)]
        previous_dim=input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim,hidden_dim))
            if batch_norm:
                layers.append(
                    nn.BatchNorm1d(hidden_dim)
                )
            layers.append(activation_class())
            if dropout>0.0:
                layers.append(
                    nn.Dropout(p=dropout)
                )
            previous_dim=hidden_dim

        layers.append(nn.Linear(previous_dim,output_dim))
        self.network=nn.Sequential(*layers)


    def forward(self,x):
        return self.network(x)



if __name__ == "__main__":
    model=MLP(input_dim=192,hidden_dims=[128,128],output_dim=10)
    print(model)

    x=torch.randn(64,3,8,8)
    output=model.forward(x)

    print(output.shape)





