import copy
import torch.nn as nn

PAD = 0 # padding占位符的索引
UNK = 1 # 未登录词标识符的索引
BATCH_SIZE = 128 # 批次大小
EPOCHS = 100 # 训练轮数
LAYERS = 6 # transformer中encoder、decoder层数 6->12
H_NUM = 8 # 多头注意力个数   8->16
D_MODEL = 256 # 输入、输出词向量维数
D_FF = 1024 # feed forward全连接层维数
DROPOUT = 0.1 # dropout比例
MAX_LENGTH = 120 # 语句最大长度

TRAIN_FILE = 'nmt/Classic-Chinese/train_small.txt' # 训练集
DEV_FILE = "nmt/Classic-Chinese/dev_small.txt" # 验证集
SAVE_FILE = 'save/model_small.pt' # 模型保存路径
DEVICE = 'cuda'

def clones(module, N):
    """ 2024.07.12
    克隆基本单元，克隆的单元之间参数不共享
    :param module: 基本单元
    :param N: 克隆数量
    :return: 克隆的单元列表
    """
    return nn.ModuleList([
        copy.deepcopy(module) for _ in range(N)
    ])