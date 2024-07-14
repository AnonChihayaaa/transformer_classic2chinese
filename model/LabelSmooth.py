from torch import nn
import torch
from torch.autograd import Variable

class LabelSmoothing(nn.Module):
    """ 2024.07.13
    Label Smoothing
    :func __init__: 初始化Label Smoothing
    :func forward: 前向传播
    """

    def __init__(self, size, padding_idx, smoothing=0.0):
        """ 2024.07.13
        :param size: 词表大小
        :param padding_idx: 填充符
        :param smoothing: 平滑值
        """
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        """ 2024.07.13
        :param x: 输入
        :param target: 目标
        :return: Loss
        """
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    # Label smoothing的例子
    crit = LabelSmoothing(5, 0, 0.4)  # 设定一个ϵ=0.4
    predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
                                 [0, 0.2, 0.7, 0.1, 0],
                                 [0, 0.2, 0.7, 0.1, 0]])
    v = crit(Variable(predict.log()),
             Variable(torch.LongTensor([2, 1, 0])))

    print(crit.true_dist)
    plt.imshow(crit.true_dist)