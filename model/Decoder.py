from LayerNorm import LayerNorm
from connection import SublayerConnection
from utils import clones
from torch import nn

class DecoderLayer(nn.Module):
    """ 2024.07.13
    解码器基本单元
    :func __init__: 初始化解码器基本单元
    :func forward: 前向传播
    """
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        """ 2024.07.13
        :param size: 特征维数
        :param self_attn: 自注意力机制
        :param src_attn: 上下文注意力机制
        :param feed_forward: 前馈神经网络
        :param dropout: dropout比例
        """
        super(DecoderLayer, self).__init__()
        self.size = size
        # 自注意力机制
        self.self_attn = self_attn
        # 上下文注意力机制
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """ 2024.07.13
        :param x: 输入
        :param memory: 编码器输出
        :param src_mask: 源掩码
        :param tgt_mask: 目标掩码
        :return: 解码器基本单元结果
        """
        # memory为编码器输出隐表示
        m = memory
        # 自注意力机制，q、k、v均来自解码器隐表示
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 上下文注意力机制：q为来自解码器隐表示，而k、v为编码器隐表示
        x = self.sublayer[1](x, lambda x: self.self_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
    
class Decoder(nn.Module):
    """ 2024.07.13
    解码器
    :func __init__: 初始化解码器
    :func forward: 前向传播
    """
    def __init__(self, layer, N):
        """ 2024.07.13
        :param layer: 解码器基本单元
        :param N: 解码器基本单元数量
        """
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """ 2024.07.13
        :param x: 输入
        :param memory: 编码器输出
        :param src_mask: 源掩码
        :param tgt_mask: 目标掩码
        :return: 解码器结果
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)