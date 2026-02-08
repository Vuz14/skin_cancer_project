import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )

    def forward(self, x):
        # Global Avg Pool + Global Max Pool
        avg_pool = self.mlp(F.avg_pool2d(x, (x.size(2), x.size(3))))
        max_pool = self.mlp(F.max_pool2d(x, (x.size(2), x.size(3))))
        channel_att_sum = avg_pool + max_pool

        # Sigmoid để tạo mask (0-1)
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        # Convolution 7x7 để bắt đặc trưng không gian
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        # Max Pool + Avg Pool theo channel
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, dim=1, keepdim=True)

        # Nối lại và qua Conv
        scale = torch.sigmoid(self.spatial(torch.cat([max_pool, avg_pool], dim=1)))
        return x * scale


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_gate = ChannelGate(channels, reduction)
        self.spatial_gate = SpatialGate()

    def forward(self, x):
        x_out = self.channel_gate(x)
        x_out = self.spatial_gate(x_out)
        return x_out