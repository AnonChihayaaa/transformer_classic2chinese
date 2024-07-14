import torch
import torch.nn as nn
import math
from torch.autograd import Variable
from utils import DEVICE

class PositionalEncoding(nn.Module):
    """ 2024.07.11
    位置编码
    :func __init__: 初始化位置编码
    :func forward: 前向传播
    """
    def __init__(self, d_model, dropout, max_len=5000):
        """ 2024.07.14
        :param d_model: 词向量维数
        :param dropout: dropout比例
        :param max_len: 最大长度
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 位置编码矩阵，维度[max_len, embedding_dim]
        pe = torch.zeros(max_len, d_model, device=DEVICE)
        # 单词位置
        position = torch.arange(0.0, max_len, device=DEVICE)
        position.unsqueeze_(1)
        # 使用exp和log实现幂运算
        div_term = torch.exp(torch.arange(0.0, d_model, 2, device=DEVICE) * (- math.log(1e4) / d_model))
        div_term.unsqueeze_(0)
        # 计算单词位置沿词向量维度的纹理值
        pe[:, 0 : : 2] = torch.sin(torch.mm(position, div_term))
        pe[:, 1 : : 2] = torch.cos(torch.mm(position, div_term))
        # 增加批次维度，[1, max_len, embedding_dim]
        pe.unsqueeze_(0)
        # 将位置编码矩阵注册为buffer(不参加训练)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """ 2024.07.11
        :param x: 输入的词向量
        :return: 词向量与位置编码相加
        """
        # 将一个批次中语句所有词向量与位置编码相加  
        x += Variable(self.pe[:, : x.size(1), :], requires_grad=False) # 位置编码不参与训练，因此设置requires_grad=False
        return self.dropout(x)
    
if __name__ == '__main__':  
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    emb_dim = 64
    max_seq_len = 100
    seq_len = 20
    pe = PositionalEncoding(emb_dim, 0, max_seq_len)
    positional_encoding = pe(torch.zeros(1, seq_len, emb_dim, device=DEVICE))
    plt.figure()
    sns.heatmap(positional_encoding.squeeze().to("cpu"))
    plt.xlabel("i")
    plt.ylabel("pos")
    plt.show()
    plt.figure()
    y = positional_encoding.to("cpu").numpy()
    plt.plot(np.arange(seq_len), y[0, :, 0 : 64 : 8], ".")
    plt.legend(["dim %d" % p for p in [0, 7, 15, 31, 63]])
    plt.show()