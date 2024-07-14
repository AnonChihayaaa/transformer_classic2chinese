import torch
import torch.nn.functional as F
from torch.autograd import Variable

import sys
import os
sys.path.append(os.path.abspath(os.path.dirname("model/")))
from dataset import subsequent_mask

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """ 2024.07.14
    传入一个训练好的模型，对指定数据进行预测
    :param model: 训练好的模型
    :param src: 输入数据
    :param src_mask: 输入数据掩码
    :param max_len: 最大长度
    :param start_symbol: 开始符
    :return: 预测结果
    """
    # 先用encoder进行encode
    memory = model.encode(src, src_mask)
    # 初始化预测内容为1×1的tensor，填入开始符('BOS')的id，并将type设置为输入数据类型(LongTensor)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    # 遍历输出的长度下标
    for i in range(max_len - 1):
        # decode得到隐层表示
        out = model.decode(memory,
                           src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        # 将隐藏表示转为对词典各词的log_softmax概率分布表示
        prob = model.generator(out[:, -1])
        # 获取当前位置最大概率的预测词id
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        # 将当前位置预测的字符id与之前的预测内容拼接起来
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

def top_k_decode(model, src, src_mask, max_len, start_symbol, k):
    """ 2024.07.14
    使用top-k作为解码策略对指定数据进行预测
    :param model: 训练好的模型
    :param src: 输入数据
    :param src_mask: 输入数据掩码
    :param max_len: 最大长度
    :param start_symbol: 开始符
    :param k: top-k的k值
    :return: 预测结果
    """
    # 先用encoder进行encode
    memory = model.encode(src, src_mask)
    # 初始化预测内容为1×1的tensor，填入开始符('BOS')的id，并将type设置为输入数据类型(LongTensor)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    # 遍历输出的长度下标
    for i in range(max_len - 1):
        # decode得到隐层表示
        out = model.decode(memory,
                           src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        # 将隐藏表示转为对词典各词的log_softmax概率分布表示
        prob = model.generator(out[:, -1])
        # 获取top-k的候选词及其概率
        top_k_prob, top_k_words = torch.topk(prob, k, dim=1)
        # 从top-k候选词中按概率分布随机选择下一个词
        top_k_prob = F.softmax(top_k_prob, dim=1)
        next_word = top_k_words[0, torch.multinomial(top_k_prob[0], 1)]
        # 将当前位置预测的字符id与之前的预测内容拼接起来
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

def top_p_filtering(logits, top_p=0.95, filter_value=-float('Inf')):
    """ 2024.07.14
    使用nucleus sampling进行解码
    :param logits: 模型输出的logits
    :param top_p: 超参数top-p   
    :param filter_value: 过滤值
    :return: 过滤后的logits
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = filter_value
    return logits

def top_p_decode(model, src, src_mask, max_len, start_symbol, top_p=0.95):
    """ 2024.07.14
    使用top-p进行解码
    :param model: 训练好的模型
    :param src: 输入数据
    :param src_mask: 输入数据掩码
    :param max_len: 最大长度
    :param start_symbol: 开始符
    :param top_p: top-p的p值
    :return: 预测结果
    """
    # 结果编码
    memory = model.encode(src, src_mask)
    # 初始化预测内容为1×1的tensor，填入开始符('BOS')的id，并将type设置为输入数据类型(LongTensor)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        # Decode得到隐层表示
        out = model.decode(memory, src_mask, Variable(ys), Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        # 将隐藏表示转为对词典各词的log_softmax概率分布表示
        logits = model.generator(out[:, -1])
        # 过滤掉低概率的词
        filtered_logits = top_p_filtering(logits, top_p=top_p)
        # 对过滤后的概率分布进行softmax归一化
        probs = F.softmax(filtered_logits, dim=-1)
        print(probs)
        # 从过滤后的概率分布中按概率分布随机选择下一个词
        next_token = torch.multinomial(probs, 1)
        next_word = next_token.item()
        # 将当前位置预测的字符id与之前的预测内容拼接起来
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys