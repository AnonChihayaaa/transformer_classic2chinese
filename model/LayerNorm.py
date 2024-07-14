import torch
from torch import nn

class LayerNorm(nn.Module):
    """ 2024.07.12
    层归一化
    :func __init__: 初始化层归一化
    :func forward: 前向传播
    """
    def __init__(self, features, eps=1e-6):
        """ 2024.07.14
        :param features: 特征维数
        :param eps: 平滑项
        """
        super(LayerNorm, self).__init__()
        # α、β分别初始化为1、0
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        # 平滑项
        self.eps = eps

    def forward(self, x):
        """ 2024.07.14
        :param x: 输入
        :return: 层归一化结果
        """
        # 沿词向量方向计算均值和方差
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        # 归一化
        x = (x - mean) / torch.sqrt(std ** 2 + self.eps)
        return self.a_2 * x + self.b_2