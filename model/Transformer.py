import sys
import os

from torch import nn
import copy

sys.path.append(os.path.abspath(os.path.dirname("model/")))

from attention import MultiHeadedAttention
from FeedForward import PositionwiseFeedForward
from Encoder import Encoder, EncoderLayer
from Decoder import Decoder, DecoderLayer
from utils import DEVICE
from PositionalEncoding import PositionalEncoding
import math
import torch.nn.functional as F

class Embeddings(nn.Module):
    """ 2024.07.13
    词嵌入层
    :func __init__: 初始化词嵌入层
    :func forward: 前向传播
    """
    def __init__(self, d_model, vocab):
        """ 2024.07.14
        :param d_model: 词向量维数
        :param vocab: 词表大小
        """
        super(Embeddings, self).__init__()
        # Embedding层
        self.lut = nn.Embedding(vocab, d_model)
        # Embedding维数
        self.d_model = d_model

    def forward(self, x):
        """ 2024.07.14
        :param x: 输入的词id
        :return: x的词向量表示
        """
        # 返回x的词向量表示
        return self.lut(x) * math.sqrt(self.d_model)

class Generator(nn.Module):
    """ 2024.07.13
    解码器输出经线性变换和softmax函数映射为下一时刻预测单词的概率分布
    :func __init__: 初始化生成器
    :func forward: 前向传播
    """
    def __init__(self, d_model, vocab):
        """ 2024.07.14
        :param d_model: 词向量维数
        :param vocab: 词表大小
        """
        super(Generator, self).__init__()
        # decode后的结果，先进入一个全连接层变为词典大小的向量
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        """ 2024.07.13
        :param x: 输入
        :return: 解码器输出
        """
        # 然后再进行log_softmax操作(在softmax结果上再做多一次log运算)
        return F.log_softmax(self.proj(x), dim=-1)

class Transformer(nn.Module):
    """ 2024.07.13
    Transformer模型
    :func __init__: 初始化Transformer模型
    :func encode: 编码
    :func decode: 解码
    :func forward: 前向传播
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        """ 2024.07.13
        :param encoder: 编码器
        :param decoder: 解码器
        :param src_embed: 输入词嵌入
        :param tgt_embed: 输出词嵌入
        :param generator: 生成器
        """
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src, src_mask):
        """ 2024.07.13
        :param src: 输入
        :param src_mask: 掩码
        :return: 编码结果
        """
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """ 2024.07.13
        :param memory: 编码结果
        :param src_mask: 源掩码
        :param tgt: 输出
        :param tgt_mask: 目标掩码
        :return: 解码结果
        """
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        """ 2024.07.13
        :param src: 输入
        :param tgt: 输出
        :param src_mask: 源掩码
        :param tgt_mask: 目标掩码
        :return: 前向传播结果
        """
        # encoder的结果作为decoder的memory参数传入，进行decode
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
    
def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """ 2024.07.13
    :param src_vocab: 输入词表大小
    :param tgt_vocab: 输出词表大小
    :param N: 编码器、解码器基本单元数量
    :param d_model: 词向量维数
    :param d_ff: 前馈神经网络维数
    :param h: 头的数量
    :param dropout: dropout比例
    :return: Transformer模型
    """
    c = copy.deepcopy
    # 实例化Attention对象
    attn = MultiHeadedAttention(h, d_model).to(DEVICE)
    # 实例化FeedForward对象
    ff = PositionwiseFeedForward(d_model, d_ff, dropout).to(DEVICE)
    # 实例化PositionalEncoding对象
    position = PositionalEncoding(d_model, dropout).to(DEVICE)
    # 实例化Transformer模型对象
    model = Transformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout).to(DEVICE), N).to(DEVICE),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout).to(DEVICE), N).to(DEVICE),
        nn.Sequential(Embeddings(d_model, src_vocab).to(DEVICE), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab).to(DEVICE), c(position)),
        Generator(d_model, tgt_vocab)).to(DEVICE)
    for p in model.parameters():
        if p.dim() > 1:
            # 这里初始化采用的是nn.init.xavier_uniform
            nn.init.xavier_uniform_(p)
    return model.to(DEVICE)