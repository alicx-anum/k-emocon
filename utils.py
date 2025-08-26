# 工具包用来集成训练和测试的代码
#原来
import os
import sys
import json
import pickle
import random
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    scaler = GradScaler()  # 在训练最开始之前实例化一个GradScaler对象
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失 {Tensor:(1,)} tensor([0.])
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数 {Tensor:(1,)} tensor([0.])
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        # 前向过程(model + loss)开启 autocast，混合精度训练
        with autocast():
            outputs = model(images.to(device))
            loss = loss_function(outputs, labels.to(device))

        # outputs = model(images.to(device))
        pred_classes = torch.max(outputs, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        # loss = loss_function(outputs, labels.to(device))
        # loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch+1,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        # optimizer.step()
        # optimizer.zero_grad()

        optimizer.zero_grad()  # 梯度清零
        scaler.scale(loss).backward()  # 梯度放大
        scaler.step(optimizer)  # unscale梯度值
        scaler.update()
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        with autocast():
            pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch+1,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
















