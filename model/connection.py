from LayerNorm import LayerNorm
from torch import nn

class SublayerConnection(nn.Module):
    """ 2024.07.13
    通过层归一化和残差连接，连接Multi-Head Attention和Feed Forward
    :func __init__: 初始化连接
    :func forward: 前向传播
    """
    def __init__(self, size, dropout):
        """ 2024.07.13
        :param size: 特征维数
        :param dropout: dropout比例
        """
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """ 2024.07.13
        :param x: 输入
        :param sublayer: 子层
        :return: 层归一化结果
        """
        # 层归一化
        x_ = self.norm(x)
        x_ = sublayer(x_)
        x_ = self.dropout(x_)
        # 残差连接
        return x + x_