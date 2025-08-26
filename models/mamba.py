import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x + residual


class Mamba(nn.Module):
    def __init__(self, input_len=1280, in_channels=2, num_classes=2, dim=64, depth=4, hidden_dim=128):
        super(Mamba, self).__init__()

        self.input_proj = nn.Conv1d(in_channels=1, out_channels=dim, kernel_size=1)
        self.blocks = nn.Sequential(*[MambaBlock(dim, hidden_dim) for _ in range(depth)])
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        # x: [batch, 1, seq_len, modal]
        x = x.squeeze(1).permute(0, 2, 1)  # [batch, modal, seq_len]
        x = self.input_proj(x)            # [batch, dim, seq_len]
        x = x.permute(0, 2, 1)            # [batch, seq_len, dim]
        x = self.blocks(x)                # [batch, seq_len, dim]
        x = x.permute(0, 2, 1)            # [batch, dim, seq_len]
        x = self.global_pool(x).squeeze(-1)  # [batch, dim]
        x = self.fc(x)                    # [batch, num_classes]
        return x
