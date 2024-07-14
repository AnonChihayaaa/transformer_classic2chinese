import torch
import time
from utils import EPOCHS, SAVE_FILE
from torch.utils.tensorboard import SummaryWriter  # 引入SummaryWriter
from model.dataset import PrepareData
from model.Transformer import make_model
from utils import *
from model.LabelSmooth import LabelSmoothing


class SimpleLossCompute:
    """ 2024.07.13
    简单的计算损失和进行参数反向传播更新训练的函数
    :func __init__: 初始化计算损失和进行参数反向传播更新训练的函数
    :func __call__: 调用函数
    """

    def __init__(self, generator, criterion, opt=None):
        """ 2024.07.13
        :param generator: 生成器
        :param criterion: 损失函数
        :param opt: 优化器
        """
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        """ 2024.07.13
        :param x: 输入
        :param y: 输出
        :param norm: 归一化
        :return: 损失值
        """
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data.item() * norm.float()


class NoamOpt:
    """ 2024.07.13
    优化器
    :func __init__: 初始化优化器
    :func step: 更新参数和速率
    :func rate: 计算学习率
    """

    def __init__(self, model_size, factor, warmup, optimizer):
        """ 2024.07.13
        :param model_size: 模型大小
        :param factor: 因子
        :param warmup: 热身
        :param optimizer: 优化器
        """
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        """ 2024.07.13
        更新参数和速率
        """
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """ 2024.07.13
        :param step: 步数
        :return: 学习率
        """
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    """ 2024.07.13
    :param model: 模型
    :return: 优化器
    """
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

def run_epoch(data, model, loss_compute, epoch):
    """ 2024.07.13
    :param data: 数据
    :param model: 模型
    :param loss_compute: 损失计算
    :param epoch: epoch
    :return: 总损失
    """
    start = time.time()
    total_tokens = 0.
    total_loss = 0.
    tokens = 0.

    for i, batch in enumerate(data):
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch %d Batch: %d Loss: %f Tokens per Sec: %fs" % (
            epoch, i - 1, loss / batch.ntokens, (tokens.float() / elapsed / 1000.)))
            start = time.time()
            tokens = 0

    return total_loss / total_tokens


def train(data, model, criterion, optimizer):
    """ 2024.07.13
    训练并保存模型
    :param data: 数据
    :param model: 模型
    :param criterion: 损失函数
    :param optimizer: 优化器
    """
    # 初始化模型在dev集上的最优Loss为一个较大值
    best_dev_loss = 1e5

    for epoch in range(EPOCHS):
        # 模型训练
        model.train()
        train_loss = run_epoch(data.train_data, model, SimpleLossCompute(model.generator, criterion, optimizer), epoch)
        model.eval()

        # 在dev集上进行loss评估
        print('>>>>> Evaluate')
        dev_loss = run_epoch(data.dev_data, model, SimpleLossCompute(model.generator, criterion, None), epoch)
        print('<<<<< Evaluate loss: %f' % dev_loss)

        # 记录训练和评估的loss值
        writer.add_scalars('loss', {'train': train_loss, 'dev': dev_loss}, epoch)

        # 如果当前epoch的模型在dev集上的loss优于之前记录的最优loss则保存当前模型，并更新最优loss值
        if dev_loss < best_dev_loss:
            torch.save(model.state_dict(), SAVE_FILE)
            best_dev_loss = dev_loss
            print('****** Save model done... ******')

        print()

    # 关闭writer
    writer.close()

if __name__ == '__main__':
    # 数据预处理
    data = PrepareData(TRAIN_FILE, DEV_FILE)
    src_vocab = len(data.en_word_dict)
    tgt_vocab = len(data.cn_word_dict)
    print("src_vocab %d" % src_vocab)
    print("tgt_vocab %d" % tgt_vocab)

    # 初始化模型
    model = make_model(
                    src_vocab,
                    tgt_vocab,
                    LAYERS,
                    D_MODEL,
                    D_FF,
                    H_NUM,
                    DROPOUT
                )
    # 训练
    print(">>>>>>> start train")
    train_start = time.time()
    criterion = LabelSmoothing(tgt_vocab, padding_idx = 0, smoothing= 0.0)
    optimizer = NoamOpt(D_MODEL, 1, 2000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9,0.98), eps=1e-9))
    # 初始化一个summary writer，保存在当前目录下的runs文件夹中
    writer = SummaryWriter()
    train(data, model, criterion, optimizer)
    print(f"<<<<<<< finished train, cost {time.time()-train_start:.4f} seconds")