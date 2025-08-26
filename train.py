import argparse
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from models import cnn, resnet, res2net, resnext, sk_resnet, resnest, lstm, dilated_conv, depthwise_conv, shufflenet, vit, dcn, channel_attention, spatial_attention, swin,add,mamba,mamba2
from utils import train_one_epoch, evaluate
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, \
    ConfusionMatrixDisplay
import time


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Emotion recognition task')
    parser.add_argument(
        '--model',
        help='select network',
        choices=model_dict.keys(),
        default='mamba'
        #cnn、resnet、depthwise、mamba
        )
    parser.add_argument('--batch', type=int, help='batch_size', default=32)
    parser.add_argument('--epoch', type=int, help='epoch', default=30)
    parser.add_argument('--lr', type=float, help='learning_rate', default=0.000001)
    args = parser.parse_args()
    return args
def main(args):
    # 检查GPU可用性
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU可用")
    else:
        device = torch.device("cpu")
        print("GPU不可用，将使用CPU")

    # =================== 修改后的读取部分 ===================
    #arousal和valence分别读取不一样的特征数据/标签数据文件，X和y都要改
    # #S_A2对应arousal
    # X = pd.read_csv('KEmoCon_two/S_A2/win5s_overlap0.50/E4_BVP_data.csv', header=None).values  # shape: [5305, 20]
    # # 标签数据:
    # y = pd.read_csv('KEmoCon_two/S_A2/win5s_overlap0.50/label_arousal.csv',
    #                 header=None).values.squeeze()  # shape: [5305]
    # # 确保标签是整数类型（如 0 和 1）
    # y = y.astype(int)

    # # S_V2对应valence
    # X = pd.read_csv('KEmoCon_two/S_V2/win5s_overlap0.50/E4_BVP_data.csv', header=None).values  # shape: [5305, 20]
    # # 标签数据:
    # y = pd.read_csv('KEmoCon_two/S_V2/win5s_overlap0.50/label_valence.csv',
    #                 header=None).values.squeeze()  # shape: [5305]
    # # 确保标签是整数类型（如 0 和 1）
    # y = y.astype(int)

    # 特征数据:每个先读取，四个一起用
    # ========= Arousal =========
    bvp = pd.read_csv('KEmoCon_two/S_A2/win5s_overlap0.50/E4_BVP_data.csv', header=None).values
    eda = pd.read_csv('KEmoCon_two/S_A2/win5s_overlap0.50/E4_EDA_data.csv', header=None).values
    hr = pd.read_csv('KEmoCon_two/S_A2/win5s_overlap0.50/E4_HR_data.csv', header=None).values
    temp = pd.read_csv('KEmoCon_two/S_A2/win5s_overlap0.50/E4_TEMP_data.csv', header=None).values
    X = np.hstack([bvp, eda, hr, temp])

    # 标签数据:
    y = pd.read_csv('KEmoCon_two/S_A2/win5s_overlap0.50/label_arousal.csv',
                    header=None).values.squeeze()  # shape: [5305]
    # 确保标签是整数类型（如 0 和 1）
    y = y.astype(int)

    # # ========= Valence =========
    # bvp = pd.read_csv('KEmoCon_two/S_V2/win5s_overlap0.50/E4_BVP_data.csv', header=None).values
    # eda = pd.read_csv('KEmoCon_two/S_V2/win5s_overlap0.50/E4_EDA_data.csv', header=None).values
    # hr = pd.read_csv('KEmoCon_two/S_V2/win5s_overlap0.50/E4_HR_data.csv', header=None).values
    # temp = pd.read_csv('KEmoCon_two/S_V2/win5s_overlap0.50/E4_TEMP_data.csv', header=None).values
    # X = np.hstack([bvp,eda,hr,temp])
    #
    # # 标签数据:
    # y = pd.read_csv('KEmoCon_two/S_V2/win5s_overlap0.50/label_valence.csv',
    #                 header=None).values.squeeze()  # shape: [5305]
    # # 确保标签是整数类型（如 0 和 1）
    # y = y.astype(int)


    # 将数据集划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 将数据转换为PyTorch张量，并移动到GPU
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device).unsqueeze(1).unsqueeze(-1)
    #维度是从0开始计数
    #unsqueeze(1)在第1维度上增加一个维度
    #例如：原始形状是[batch_size, features]，操作后会变成[batch_size, 1, features]
    #unsqueeze(-1)在最后一个维度上增加一个维度
    #例如：形状从 [batch_size, 1, features] 变为 [batch_size, 1, features, 1]
    y_train = torch.tensor(y_train, dtype=torch.int64).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device).unsqueeze(1).unsqueeze(-1)
    y_test = torch.tensor(y_test, dtype=torch.int64).to(device)

    print(X_test.shape)
    print(X_train.shape)
    print(y_test.shape)
    print(y_train.shape)
    '''
    torch.Size([1061, 1, 320, 1])
    torch.Size([4244, 1, 320, 1])
    torch.Size([1061])
    torch.Size([4244])
    '''

    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    # 创建测试数据加载器
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)


    model = model_dict[args.model](X_train.shape, 2).to(device)

    print(model)

    #查看是否正确

    # 假设你已经有 X_train, y_train, model, device
    # X_train shape: [num_samples, 1, features, 1]
    # y_train shape: [num_samples]

    # 创建数据加载器
    batch_size = 8
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 取一个 batch
    batch_x, batch_y = next(iter(train_loader))
    print("=== 数据维度检查 ===")
    print("batch_x shape:", batch_x.shape)
    print("batch_y shape:", batch_y.shape)
    print("batch_y dtype:", batch_y.dtype)
    print("batch_y example:", batch_y[:5])

    # 模型 forward
    model.eval()
    with torch.no_grad():
        outputs = model(batch_x)
    print("\n=== 模型输出检查 ===")
    print("outputs shape:", outputs.shape)
    print("outputs example:", outputs[:5])

    # 根据输出形状判断预测方式
    if outputs.shape[1] == 2:
        preds = torch.argmax(outputs, dim=1)
    elif outputs.shape[1] == 1:
        preds = torch.sigmoid(outputs).round()
    else:
        print("警告：输出形状不标准！")
        preds = None

    if preds is not None:
        print("\n=== 预测检查 ===")
        print("preds shape:", preds.shape)
        print("preds example:", preds[:5])

    # 检查预测是否单一类别
    if preds is not None:
        unique_classes = torch.unique(preds)
        print("预测类别分布:", unique_classes)

    epochs = args.epoch

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.001)
    scheduler = StepLR(optimizer, step_size=epochs//3, gamma=0.5)  # 每epochs//3个 epoch 学习率减小为原来的50%

    # 定义列表来保存训练过程中的损失和准确率
    train_loss_all = []
    train_acc_all = []
    test_loss_all = []
    test_acc_all = []

    print('\n==================================================   【训练】   ===================================================\n')
    for epoch in range(epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)
        scheduler.step()

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=test_loader,
                                     device=device,
                                     epoch=epoch)
        # 保存训练过程中的损失和准确率
        train_loss_all.append(train_loss)
        train_acc_all.append(train_acc)
        # 保存测试过程中的损失和准确率
        test_loss_all.append(val_loss)
        test_acc_all.append(val_acc)

    print("over")


    # 绘制训练过程中的损失和准确率曲线
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs + 1), train_loss_all, label='Train Loss')
    plt.plot(range(1, epochs + 1), test_loss_all, label='Test Loss')
    plt.plot(range(1, epochs + 1), train_acc_all, label='Train Accuracy')
    plt.plot(range(1, epochs + 1), test_acc_all, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('value')
    plt.legend()
    plt.show()

    print(
        '\n==================================================   【测试评估指标】   ===================================================\n')

    # 模型评估阶段不需要梯度
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    # 计算评估指标
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')

    print(f"准确率 Accuracy: {acc:.4f}")
    print(f"精确率 Precision: {precision:.4f}")
    print(f"召回率 Recall: {recall:.4f}")
    print(f"F1 分数: {f1:.4f}")

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    # 参数量统计
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")

    # ========================== 推理时间统计 ==========================
    with torch.no_grad():
        # 单样本推理时间（取一个样本多次前向传播）
        single_sample = X_test[0].unsqueeze(0)  # shape: [1, 1, series, modal]
        repetitions = 100
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        for _ in range(repetitions):
            _ = model(single_sample)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        avg_sample_time = (end_time - start_time) / repetitions
        print(f"单样本平均推理时间: {avg_sample_time:.4f} 秒")

        # 整个测试集一轮推理时间
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        for batch_x, _ in test_loader:
            _ = model(batch_x)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        total_test_time = end_time - start_time
        print(f"整个测试集一轮推理时间: {total_test_time:.4f} 秒（总样本数: {len(test_dataset)}）")


if __name__ == '__main__':
    model_dict = {
        'cnn': cnn.CNN,
        'resnet': resnet.ResNet,
        'res2net': res2net.Res2Net,
        'resnext': resnext.ResNext,
        'sknet': sk_resnet.SKResNet,
        'resnest': resnest.ResNeSt,
        'lstm': lstm.LSTM,
        'ca': channel_attention.ChannelAttentionNeuralNetwork,
        'sa': spatial_attention.SpatialAttentionNeuralNetwork,
        'dilation': dilated_conv.DilatedConv,
        'depthwise': depthwise_conv.DepthwiseConv,
        'shufflenet': shufflenet.ShuffleNet,
        'dcn': dcn.DeformableConvolutionalNetwork,
        'vit': vit.VisionTransformer,
        'swin': swin.SwinTransformer,
        'add':add.HybridModel,
        'mamba':mamba.Mamba,
        'mam2':mamba2.Mamba
    }
    opt = parse_args()
    main(opt)












