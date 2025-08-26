import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLSTM(nn.Module):
    """LSTM with Attention Mechanism"""

    def __init__(self, input_dim, hidden_dim):
        super(AttentionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_dim]
        attention_weights = self.attention(lstm_out)  # [batch, seq_len, 1]
        context = torch.sum(attention_weights * lstm_out, dim=1)  # [batch, hidden_dim]
        return context


class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation Block"""

    def __init__(self, channel, reduction=16):
        super(SqueezeExcite, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


class HybridModel(nn.Module):
    def __init__(self, input_shape, num_classes, lstm_units=384, conv_filters=[128, 256, 128]):
        super(HybridModel, self).__init__()
        max_nb_vars, max_timesteps = input_shape[1], input_shape[2]
        stride = 3  # Subsampling factor

        # Temporal path (Attention LSTM)
        self.temporal_conv = nn.Conv1d(max_timesteps, max_nb_vars // stride, kernel_size=8,
                                       stride=stride, padding=3, bias=False)
        self.lstm = AttentionLSTM(max_nb_vars // stride, lstm_units)
        self.temporal_dropout = nn.Dropout(0.8)

        # Spatial path (Conv1D + SE)
        self.spatial_conv1 = nn.Sequential(
            nn.Conv1d(max_nb_vars, conv_filters[0], kernel_size=8, padding='same'),
            nn.BatchNorm1d(conv_filters[0]),
            nn.ReLU()
        )
        self.se1 = SqueezeExcite(conv_filters[0])

        self.spatial_conv2 = nn.Sequential(
            nn.Conv1d(conv_filters[0], conv_filters[1], kernel_size=5, padding='same'),
            nn.BatchNorm1d(conv_filters[1]),
            nn.ReLU()
        )
        self.se2 = SqueezeExcite(conv_filters[1])

        self.spatial_conv3 = nn.Sequential(
            nn.Conv1d(conv_filters[1], conv_filters[2], kernel_size=3, padding='same'),
            nn.BatchNorm1d(conv_filters[2]),
            nn.ReLU()
        )
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Final classifier
        self.fc = nn.Linear(lstm_units + conv_filters[2], num_classes)

        # Initialize weights
        self._init_weights()

    def forward(self, x):
        # Input shape: [batch, max_nb_vars, max_timesteps]

        # Temporal path
        x_temp = x.permute(0, 2, 1)  # [batch, timesteps, vars]
        x_temp = self.temporal_conv(x_temp)  # [batch, vars//stride, timesteps]
        x_temp = x_temp.permute(0, 2, 1)  # [batch, timesteps, vars//stride]
        x_temp = self.lstm(x_temp)
        x_temp = self.temporal_dropout(x_temp)

        # Spatial path
        x_spatial = x  # [batch, vars, timesteps]
        x_spatial = self.spatial_conv1(x_spatial)
        x_spatial = self.se1(x_spatial)
        x_spatial = self.spatial_conv2(x_spatial)
        x_spatial = self.se2(x_spatial)
        x_spatial = self.spatial_conv3(x_spatial)
        x_spatial = self.gap(x_spatial).squeeze(-1)

        # Concatenate features
        x = torch.cat([x_temp, x_spatial], dim=1)
        x = self.fc(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)