import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class PHMLinear(nn.Module):
    def __init__(self, n, in_features, out_features, cuda=True):
        super(PHMLinear, self).__init__()
        self.n = n
        self.in_features = in_features
        self.out_features = out_features
        self.cuda = cuda

        self.bias = nn.Parameter(torch.Tensor(out_features))

        # 初始化为零，后面调用reset_parameters会初始化
        self.A = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros((n, n, n))))
        self.S = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros((n, out_features // n, in_features // n))))

        self.weight = None

        self.reset_parameters()

    def kronecker_product1(self, a, b):  # adapted from Bayer Research's implementation
        siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
        res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
        siz0 = res.shape[:-4]
        out = res.reshape(siz0 + siz1)
        return out

    def forward(self, input):
        self.weight = torch.sum(self.kronecker_product1(self.A, self.S), dim=0)
        input = input.type(dtype=self.weight.dtype)
        return F.linear(input, weight=self.weight, bias=self.bias)

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.A, a=math.sqrt(5))
        init.kaiming_uniform_(self.S, a=math.sqrt(5))
        fan_in = self.in_features
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


class BVPBase(nn.Module):
    def __init__(self, n=4, in_features=320, units=256):
        super(BVPBase, self).__init__()
        self.linear1 = PHMLinear(n, in_features, units)
        self.bn1 = nn.BatchNorm1d(units)
        self.linear2 = PHMLinear(n, units, units // 2)
        self.bn2 = nn.BatchNorm1d(units // 2)

    def forward(self, x):
        # x: [batch, seq_len], PHMLinear输入是二维张量 [batch, features]
        x = self.linear1(x)
        x = F.relu(self.bn1(x))
        x = self.linear2(x)
        x = F.relu(self.bn2(x))
        return x


class H2SingleBVP(nn.Module):
    def __init__(self, dropout_rate=0.5, units=256, n=4, in_features=320, num_classes=2):
        super(H2SingleBVP, self).__init__()
        self.bvp_base = BVPBase(n=n, in_features=in_features, units=units)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(units // 2, num_classes)

    def forward(self, x):
        # 输入形状: [batch, 1, seq_len, 1] -> reshape成 [batch, seq_len]
        x = x.squeeze(1).squeeze(-1)
        x = self.bvp_base(x)
        x = self.dropout(x)
        out = self.classifier(x)
        return out
