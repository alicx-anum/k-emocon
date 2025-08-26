import argparse
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight  # 🔥 修改
from models import cnn, resnet, res2net, resnext, sk_resnet, resnest, lstm, dilated_conv, depthwise_conv, shufflenet, vit, dcn, channel_attention, spatial_attention, swin, add, mamba, mamba2
from mutils import train_one_epoch, evaluate
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import time
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Emotion recognition task')
    parser.add_argument('--model', help='select network', choices=model_dict.keys(), default='mamba')
    parser.add_argument('--batch', type=int, help='batch_size', default=128)
    parser.add_argument('--epoch', type=int, help='epoch', default=100)
    parser.add_argument('--lr', type=float, help='learning_rate', default=0.0001)  # 🔥 修改
    args = parser.parse_args()
    return args


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ================= 数据读取 =================
    #S_A2对应arousal
    X = pd.read_csv('KEmoCon_two/S_A2/win5s_overlap0.50/E4_BVP_data.csv', header=None).values  # shape: [5305, 20]
    # 标签数据:
    y = pd.read_csv('KEmoCon_two/S_A2/win5s_overlap0.50/label_arousal.csv',
                    header=None).values.squeeze()  # shape: [5305]
    # 确保标签是整数类型（如 0 和 1）
    y = y.astype(int)
    # ========= Valence =========
    # # S_V2对应valence
    # X = pd.read_csv('KEmoCon_two/S_V2/win5s_overlap0.50/E4_BVP_data.csv', header=None).values  # shape: [5305, 20]
    # # 标签数据:
    # y = pd.read_csv('KEmoCon_two/S_V2/win5s_overlap0.50/label_valence.csv',
    #                 header=None).values.squeeze()  # shape: [5305]
    # # 确保标签是整数类型（如 0 和 1）
    # y = y.astype(int)
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

    # #====Arousal=========
    # bvp = pd.read_csv('KEmoCon_two/S_A2/win5s_overlap0.50/E4_BVP_data.csv', header=None).values
    # eda = pd.read_csv('KEmoCon_two/S_A2/win5s_overlap0.50/E4_EDA_data.csv', header=None).values
    # hr = pd.read_csv('KEmoCon_two/S_A2/win5s_overlap0.50/E4_HR_data.csv', header=None).values
    # temp = pd.read_csv('KEmoCon_two/S_A2/win5s_overlap0.50/E4_TEMP_data.csv', header=None).values
    # X = np.hstack([bvp, eda, hr, temp])
    #
    # y = pd.read_csv('KEmoCon_two/S_A2/win5s_overlap0.50/label_arousal.csv', header=None).values.squeeze().astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)  # 🔥 修改：保证训练集和测试集比例一致

    # 转为 tensor
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device).unsqueeze(1).unsqueeze(-1)
    y_train = torch.tensor(y_train, dtype=torch.int64).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device).unsqueeze(1).unsqueeze(-1)
    y_test = torch.tensor(y_test, dtype=torch.int64).to(device)

    # 数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)

    # 模型
    model = model_dict[args.model](X_train.shape, 2).to(device)
    print(model)

    # ================= 分类权重 =================
    y_train_np = y_train.cpu().numpy()
    classes = np.array([0, 1])  # 🔥 明确二分类所有类别
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_np)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)  # 🔥 修改

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.001)
    scheduler = StepLR(optimizer, step_size=args.epoch // 3, gamma=0.5)

    train_loss_all, train_acc_all = [], []
    test_loss_all, test_acc_all = [], []

    # ================= 训练 =================
    for epoch in range(args.epoch):
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                criterion=criterion,
                                                epoch=epoch)
        scheduler.step()
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=test_loader,
                                     device=device,
                                     criterion=criterion,
                                     epoch=epoch)
        train_loss_all.append(train_loss)
        train_acc_all.append(train_acc)
        test_loss_all.append(val_loss)
        test_acc_all.append(val_acc)

    # ================= 绘图 =================
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, args.epoch + 1), train_loss_all, label='Train Loss')
    plt.plot(range(1, args.epoch + 1), test_loss_all, label='Test Loss')
    plt.plot(range(1, args.epoch + 1), train_acc_all, label='Train Accuracy')
    plt.plot(range(1, args.epoch + 1), test_acc_all, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('value')
    plt.legend()
    plt.show()

    # ================= 测试评估 =================
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
    disp.plot(cmap=plt.cm.Blues)
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
        'add': add.HybridModel,
        'mamba': mamba.Mamba,
        'mam2': mamba2.Mamba
    }
    opt = parse_args()
    main(opt)
