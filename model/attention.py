from torch import nn
import torch
import math
import torch.nn.functional as F
from utils import clones

def attention(query, key, value, mask=None, dropout=None):
    """ 2024.07.13
    Scaled Dot-Product Attention
    :param query: 查询
    :param key: 键
    :param value: 值
    :param mask: 掩码
    :param dropout: dropout比例
    :return: 加权值、注意力矩阵
    """
    # q、k、v向量长度为d_k
    d_k = query.size(-1)
    # 矩阵乘法实现q、k点积注意力，sqrt(d_k)归一化
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 注意力掩码机制
    if mask is not None:
        scores = scores.masked_fill(mask==0, -1e9)
    # 注意力矩阵softmax归一化
    p_attn = F.softmax(scores, dim=-1)
    # dropout
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 注意力对v加权
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    """ 2024.07.13
    多头注意力机制
    :func __init__: 初始化多头注意力机制
    :func forward: 前向传播
    """
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        """ 2024.07.13
        :param h: 头的数量
        :param d_model: 词向量维数
        :param dropout: dropout比例
        """
        # 确保整除
        assert d_model % h == 0
        # q、k、v向量维数
        self.d_k = d_model // h
        # 头的数量
        self.h = h
        # WQ、WK、WV矩阵及多头注意力拼接变换矩阵WO
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """ 2024.07.13
        :param query: 查询
        :param key: 键
        :param value: 值
        :param mask: 掩码
        :return: 多头注意力加权拼接结果
        """
        if mask is not None:
            mask = mask.unsqueeze(1)
        # 批次大小
        nbatches = query.size(0)
        # WQ、WK、WV分别对词向量线性变换，并将结果拆成h块
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]
        # 注意力加权
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 多头注意力加权拼接
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # 对多头注意力加权拼接结果线性变换
        return self.linears[-1](x)