import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, \
    ConfusionMatrixDisplay
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据集类，假设ppg.csv已标准化，arousal.csv是0/1标签
class PPGArousalDataset(Dataset):
    def __init__(self, ppg_csv_path, label_csv_path):
        self.X = pd.read_csv(ppg_csv_path).values.astype(np.float32)  # (N, seq_len)
        self.y = pd.read_csv(label_csv_path).values.squeeze().astype(np.int64)  # (N,)

        # 转成 (N, seq_len, 1) 模型输入格式
        self.X = self.X[:, :, None]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

# PositionalEncoding 和 Transformer 代码照抄，略微简化
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super().__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(input_size, num_channels, kernel_size, padding=(kernel_size-1)//2),
            nn.BatchNorm1d(num_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels, output_size, kernel_size, padding=(kernel_size-1)//2),
            nn.BatchNorm1d(output_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        x = x.transpose(1, 2)  # (batch, input_size, seq_len)
        y = self.tcn(x)
        return y.transpose(1, 2)  # (batch, seq_len, output_size)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.d_model = d_model

    def forward(self, src):
        # src shape: (batch, seq_len, d_model)
        src = src.transpose(0, 1)  # (seq_len, batch, d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output.transpose(0, 1)  # (batch, seq_len, d_model)

# class EmotionRecognitionModel(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super().__init__()
#         self.tcn = TCN(input_size, hidden_size, 32, 6, 0.1)  # 通道数减半
#         self.transformer = TransformerModel(hidden_size, hidden_size, 2, 1, 64, 0.1)  # 头数和层数减半，前馈层宽度减半
#         self.classifier = nn.Linear(hidden_size, num_classes)
#
#     def forward(self, x):
#         x = self.tcn(x)
#         x = self.transformer(x)
#         x = x.mean(dim=1)
#         return self.classifier(x)
class EmotionRecognitionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.tcn = TCN(input_size, hidden_size, 64, 6, 0.1)
        self.transformer = TransformerModel(hidden_size, hidden_size, 4, 2, 128, 0.1)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = self.tcn(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.classifier(x)

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0
    all_preds = []
    all_labels = []
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = correct / total
    avg_loss = running_loss / total
    return avg_loss, acc, all_labels, all_preds
def main():
    dataset = PPGArousalDataset('KEmoCon_two/S_V2/win5s_overlap0.50/E4_BVP_data.csv', 'KEmoCon_two/S_V2/win5s_overlap0.50/label_valence.csv')
    input_size = dataset.X.shape[2]

    train_idx, test_idx = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=42, stratify=dataset.y)
    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = EmotionRecognitionModel(input_size=input_size, hidden_size=64, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)  # 如果你需要调度器

    epochs = 30

    train_loss_all = []
    train_acc_all = []
    test_loss_all = []
    test_acc_all = []

    print('\n==================================================   【训练】   ===================================================\n')
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        val_loss, val_acc, all_labels, all_preds = evaluate(model, test_loader, criterion, device)

        train_loss_all.append(train_loss)
        train_acc_all.append(train_acc)
        test_loss_all.append(val_loss)
        test_acc_all.append(val_acc)

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Test Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

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

    print('\n==================================================   【测试评估指标】   ===================================================\n')

    # 评估阶段已经拿到最后一个epoch的 all_labels 和 all_preds
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    acc = accuracy_score(all_labels, all_preds)

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

    # 推理时间统计
    with torch.no_grad():
        single_sample = dataset[0][0].unsqueeze(0).to(device)  # 取第一个样本
        repetitions = 100
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        for _ in range(repetitions):
            _ = model(single_sample)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        avg_sample_time = (end_time - start_time) / repetitions
        print(f"单样本平均推理时间: {avg_sample_time:.4f} 秒")

        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            _ = model(batch_x)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        total_test_time = end_time - start_time
        print(f"整个测试集一轮推理时间: {total_test_time:.4f} 秒（总样本数: {len(test_dataset)})")

if __name__ == '__main__':
    main()

# def main():
#     dataset = PPGArousalDataset('KEmoCon_two/S_A2/win5s_overlap0.50/E4_BVP_data.csv', 'KEmoCon_two/S_A2/win5s_overlap0.50/label_arousal.csv')
#     input_size = dataset.X.shape[2]
#
#     train_idx, test_idx = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=42, stratify=dataset.y)
#     train_dataset = Subset(dataset, train_idx)
#     test_dataset = Subset(dataset, test_idx)
#
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
#
#     model = EmotionRecognitionModel(input_size=input_size, hidden_size=64, num_classes=2).to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=1e-3)
#
#     epochs = 100
#     train_losses, train_accs = [], []
#     test_losses, test_accs = [], []
#
#     for epoch in range(epochs):
#         train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
#         test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
#
#         train_losses.append(train_loss)
#         train_accs.append(train_acc)
#         test_losses.append(test_loss)
#         test_accs.append(test_acc)
#
#         print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")
#
#     # 计算并打印测试集的精准率和F1分数
#     precision = precision_score(y_true, y_pred, zero_division=0)
#     f1 = f1_score(y_true, y_pred, zero_division=0)
#     print(f"Test Precision: {precision:.4f}")
#     print(f"Test F1 Score: {f1:.4f}")
#
#     # 绘制训练曲线
#     plt.figure(figsize=(8, 6))
#     plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
#     plt.plot(range(1, epochs + 1), test_losses, label='Test Loss')
#     plt.plot(range(1, epochs + 1), train_accs, label='Train Accuracy')
#     plt.plot(range(1, epochs + 1), test_accs, label='Test Accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Value')
#     plt.title('Training and Testing Loss/Accuracy')
#     plt.legend()
#     plt.show()
#
# if __name__ == '__main__':
#     main()
