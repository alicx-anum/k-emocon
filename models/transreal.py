import torch
import torch.nn as nn
import math

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
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   batch_first=True)  # 使用batch_first=True更方便
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.d_model = d_model

    def forward(self, src):
        # src shape: (batch, seq_len, d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output  # (batch, seq_len, d_model)

class EmotionRecognitionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.tcn = TCN(input_size=input_size,
                       output_size=hidden_size,
                       num_channels=64,
                       kernel_size=6,
                       dropout=0.1)
        self.transformer = TransformerModel(input_dim=hidden_size,
                                            d_model=hidden_size,
                                            nhead=4,
                                            num_layers=2,
                                            dim_feedforward=128,
                                            dropout=0.1)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = self.tcn(x)              # (batch, seq_len, hidden_size)
        x = self.transformer(x)      # (batch, seq_len, hidden_size)
        x = x.mean(dim=1)            # 全局平均池化 (batch, hidden_size)
        return self.classifier(x)
