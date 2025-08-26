import os
import sys
import torch
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast


def train_one_epoch(model, optimizer, data_loader, device, epoch, criterion=None):
    scaler = GradScaler()
    model.train()
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        with autocast():
            outputs = model(images.to(device))
            loss = criterion(outputs, labels.to(device))

        pred_classes = torch.max(outputs, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch+1,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, criterion=None):
    model.eval()
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()
    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)
    sample_num = 0

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        with autocast():
            pred = model(images.to(device))
            loss = criterion(pred, labels.to(device))

        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch+1,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num
        )

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
