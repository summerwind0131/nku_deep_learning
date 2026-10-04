import torch
from torch import nn

def train_one_epoch(model,loader,criterion,optimizer,device,l2=0.0):
    model.train()

    data_loss_sum=0.0
    total_loss_sum=0.0
    sample_count=0

    linear_weights=[
        layer.weight for layer in model.modules() if isinstance(layer,nn.Linear)
    ]

    for images,labels in loader:
        images=images.to(device)
        labels=labels.to(device)

        optimizer.zero_grad()

        logits=model(images)
        data_loss=criterion(logits,labels)

        total_loss=data_loss

        if l2>0:
            weight_square_sum=sum(
                weight.square().sum()
                for weight in linear_weights
            )
            total_loss=data_loss+0.5*l2*weight_square_sum
        total_loss.backward()
        optimizer.step()

        batch_size=images.size(0)
        data_loss_sum+=data_loss.item()*batch_size
        total_loss_sum+=total_loss.item()*batch_size
        sample_count+=batch_size

    return (
        data_loss_sum/sample_count,
        total_loss_sum/sample_count
    )


def evaluate(model,loader,criterion,device):
    model.eval()
    loss_sum=0.0
    wrong_count=0
    sample_count=0

    with torch.no_grad():
        for images,labels in loader:
            images=images.to(device)
            labels=labels.to(device)

            logits=model(images)
            loss=criterion(logits,labels)

            predictions=logits.argmax(dim=1)
            wrong_count+=(predictions!=labels).sum().item()

            batch_size=images.size(0)
            loss_sum+=loss.item()*batch_size
            sample_count+=batch_size

        avg_loss=loss_sum/sample_count
        error_rate=wrong_count/sample_count
    return avg_loss,error_rate




