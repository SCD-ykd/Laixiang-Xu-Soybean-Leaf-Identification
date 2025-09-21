import torch
import torch.nn as nn

class DyT(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5, eps=1e-5):
        super().__init__()
        self.eps = eps  # ✅ 初始化
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        # 1. 标准化（沿特征维度）
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x = (x - mean) / (std + self.eps)
        # 2. 动态非线性变换
        x = torch.tanh(self.alpha * x)
        # 3. 可学习的缩放和偏移
        return x * self.weight + self.bias