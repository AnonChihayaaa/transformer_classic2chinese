from torch import nn
import torch.nn.functional as F

class PositionwiseFeedForward(nn.Module):
    """ 2024.07.13
    前馈神经网络
    :func __init__: 初始化前馈神经网络
    :func forward: 前向传播
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        """ 2024.07.14
        :param d_model: 词向量维数
        :param d_ff: 全连接层维数
        :param dropout: dropout比例
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """ 2024.07.13
        :param x: 输入
        :return: 前馈神经网络结果
        """
        x = self.w_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.w_2(x)
        return x