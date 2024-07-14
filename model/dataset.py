from utils import *
from collections import Counter
import numpy as np
import torch
from torch.autograd import Variable

def seq_padding(X, padding=PAD):
    """ 2024.07.11
    按批次（batch）对数据填充、长度对齐
    :param X: 输入语句数据
    :param padding: 填充符
    :return: 填充后的语句数据
    """
    # 计算该批次各条样本语句长度
    L = [len(x) for x in X]
    # 获取该批次样本中语句长度最大值
    ML = max(L)
    # 遍历该批次样本，如果语句长度小于最大长度，则用padding填充
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])

def subsequent_mask(size):
    """ 2024.07.11
    生成一个右上角(不含主对角线)为全1，左下角(含主对角线)为全0的subsequent_mask矩阵
    :param size: 矩阵大小
    :return: subsequent_mask矩阵
    """
    # 设定subsequent_mask矩阵的shape
    attn_shape = (1, size, size)
    # 生成一个右上角(不含主对角线)为全1，左下角(含主对角线)为全0的subsequent_mask矩阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # 返回一个右上角(不含主对角线)为全False，左下角(含主对角线)为全True的subsequent_mask矩阵
    return torch.from_numpy(subsequent_mask) == 0

class Batch:
    """ 2024.07.11
    批次类
    :func __init__: 初始化批次
    :func make_std_mask: 掩码操作
    """

    def __init__(self, src, trg=None, pad=PAD):
        """ 2024.07.11
        :param src: 输入
        :param trg: 输出
        :param pad: 填充符
        """
        # 将输入、输出单词id表示的数据规范成整数类型
        src = torch.from_numpy(src).to(DEVICE).long()
        trg = torch.from_numpy(trg).to(DEVICE).long()
        self.src = src
        # 对于当前输入的语句非空部分进行判断，bool序列
        # 并在seq length前面增加一维，形成维度为 1×seq length 的矩阵
        self.src_mask = (src != pad).unsqueeze(-2)
        # 如果输出目标不为空，则需要对解码器使用的目标语句进行掩码
        if trg is not None:
            # 解码器使用的目标输入部分
            self.trg = trg[:, : -1] # 去除最后一列
            # 解码器训练时应预测输出的目标结果
            self.trg_y = trg[:, 1:] #去除第一列的SOS
            # 将目标输入部分进行注意力掩码
            self.trg_mask = self.make_std_mask(self.trg, pad)
            # 将应输出的目标结果中实际的词数进行统计
            self.ntokens = (self.trg_y != pad).data.sum()

    # 掩码操作
    @staticmethod
    def make_std_mask(tgt, pad):
        """ 2024.07.11
        :param tgt: 目标
        :param pad: 填充符
        :return: 掩码
        """
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

class PrepareData:
    """ 2024.07.11
    数据预处理类
    :func __init__: 初始化数据
    :func load_data: 读取数据、分词
    :func build_dict: 构建词表
    :func word2id: 单词映射为索引
    :func split_batch: 划分批次、填充、掩码
    """
    def __init__(self, train_file, dev_file):
        """ 2024.07.11
        :param train_file: 训练集文件路径
        :param dev_file: 验证集文件路径
        """
        # 读取数据、分词
        self.train_en, self.train_cn = self.load_data(train_file)
        self.dev_en, self.dev_cn = self.load_data(dev_file)
        # 构建词表
        self.en_word_dict, self.en_total_words, self.en_index_dict = self.build_dict(self.train_en)
        self.cn_word_dict, self.cn_total_words, self.cn_index_dict = self.build_dict(self.train_cn)
        # 单词映射为索引
        self.train_en, self.train_cn = self.word2id(self.train_en, self.train_cn, self.en_word_dict, self.cn_word_dict)
        self.dev_en, self.dev_cn = self.word2id(self.dev_en, self.dev_cn, self.en_word_dict, self.cn_word_dict)
        # 划分批次、填充、掩码
        self.train_data = self.split_batch(self.train_en, self.train_cn, BATCH_SIZE)
        self.dev_data = self.split_batch(self.dev_en, self.dev_cn, BATCH_SIZE)

    def load_data(self, path):
        """ 2024.07.11
        读取古文、现代文数据、对每条样本分词并构建包含起始符和终止符的单词列表
        :param path: 文件路径
        EOS: End of sentence
        BOS: Begin of sentence
        :return: 古文、现代文数据
        """
        guwen = []
        baizhi = []
        with open(path, mode="r", encoding="utf-8") as f:
            for line in f.readlines():
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    sent_guwen, sent_baizhi = parts
                    sent_guwen = ["BOS"] + [char for char in sent_guwen.strip()] + ["EOS"]
                    sent_baizhi = ["BOS"] + [char for char in sent_baizhi.strip()] + ["EOS"]
                    guwen.append(sent_guwen)
                    baizhi.append(sent_baizhi)
                else:
                    print(f"Skipping line due to unexpected format: {line.strip()}")
        return guwen, baizhi

    def build_dict(self, sentences, max_words=5e4):
        """ 2024.07.11
        构造分词后的列表数据，构建单词-索引映射（key为单词，value为id值） 
        :param sentences: 分词后的列表数据
        :param max_words: 词表最大单词数
        UNK: 未记录的词
        PAD: padding填充符
        :return: 单词-索引映射、词表大小、索引-单词映射
        """
        # 统计数据集中单词词频
        word_count = Counter([word for sent in sentences for word in sent])
        # 按词频保留前max_words个单词构建词典
        # 添加UNK和PAD两个单词
        ls = word_count.most_common(int(max_words))
        total_words = len(ls) + 2
        word_dict = {w[0]: index + 2 for index, w in enumerate(ls)}
        word_dict['UNK'] = UNK
        word_dict['PAD'] = PAD
        # 构建id2word映射
        index_dict = {v: k for k, v in word_dict.items()}
        return word_dict, total_words, index_dict

    def word2id(self, en, cn, en_dict, cn_dict, sort=True):
        """ 2024.07.11
        将英文、中文单词列表转为单词索引列表 
        :param en: 英文单词列表
        :param cn: 中文单词列表
        :param en_dict: 英文单词-索引映射
        :param cn_dict: 中文单词-索引映射
        :param sort: 是否按照语句长度排序
        :return: 英文、中文单词索引列表
        """
        length = len(en)
        # 单词映射为索引
        out_en_ids = [[en_dict.get(word, UNK) for word in sent] for sent in en]
        out_cn_ids = [[cn_dict.get(word, UNK) for word in sent] for sent in cn]
        # 按照语句长度排序
        def len_argsort(seq):
            return sorted(range(len(seq)), key=lambda x: len(seq[x]))
        # 按相同顺序对中文、英文样本排序
        if sort:
            # 以英文语句长度排序
            sorted_index = len_argsort(out_en_ids)
            out_en_ids = [out_en_ids[idx] for idx in sorted_index]
            out_cn_ids = [out_cn_ids[idx] for idx in sorted_index]
        return out_en_ids, out_cn_ids

    def split_batch(self, en, cn, batch_size, shuffle=True):
        """ 2024.07.11
        划分批次
        :param en: 英文句子列表
        :param cn: 中文句子列表
        :param batch_size: 批次大小
        :param shuffle: 是否打乱顺序
        :return: 批次列表
        """
        # 每隔batch_size取一个索引作为后续batch的起始索引
        idx_list = np.arange(0, len(en), batch_size)
        # 起始索引随机打乱
        if shuffle:
            np.random.shuffle(idx_list)
        # 存放所有批次的语句索引
        batch_indexs = []
        for idx in idx_list:
            # 起始索引最大的批次可能发生越界，要限定其索引
            batch_indexs.append(np.arange(idx, min(idx + batch_size, len(en))))
        # 构建批次列表
        batches = []
        for batch_index in batch_indexs:
            # 按当前批次的样本索引采样
            batch_en = [en[index] for index in batch_index]
            batch_cn = [cn[index] for index in batch_index]
            # 对当前批次中所有语句填充、对齐长度
            # 维度为：batch_size * 当前批次中语句的最大长度
            batch_cn = seq_padding(batch_cn)
            batch_en = seq_padding(batch_en)
            # 将当前批次添加到批次列表
            # Batch类用于实现注意力掩码
            batches.append(Batch(batch_en, batch_cn))
        return batches