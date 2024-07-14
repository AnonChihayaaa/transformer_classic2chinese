from torch import nn
from utils import clones
from LayerNorm import LayerNorm
from connection import SublayerConnection

class EncoderLayer(nn.Module):
    """ 2024.07.13
    编码器基本单元
    :func __init__: 初始化编码器基本单元
    :func forward: 前向传播
    """
    def __init__(self, size, self_attn, feed_forward, dropout):
        """ 2024.07.13
        :param size: 特征维数
        :param self_attn: 自注意力机制
        :param feed_forward: 前馈神经网络
        :param dropout: dropout比例
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # SublayerConnection作用连接multi和ffn
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        # d_model
        self.size = size

    def forward(self, x, mask):
        """ 2024.07.13
        :param x: 输入
        :param mask: 掩码
        :return: 编码器基本单元结果
        """
        # 将embedding层进行Multi head Attention
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # attn的结果直接作为下一层输入
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    """ 2024.07.13
    编码器
    :func __init__: 初始化编码器
    :func forward: 前向传播
    """
    def __init__(self, layer, N):
        """ 2024.07.13
        :param layer: 编码器基本单元
        :param N: 编码器基本单元数量
        """
        super(Encoder, self).__init__()
        # 复制N个编码器基本单元
        self.layers = clones(layer, N)
        # 层归一化
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """ 2024.07.13
        :param x: 输入
        :param mask: 掩码
        :return: 编码器结果
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)