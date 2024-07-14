import torch
import numpy as np
import sacrebleu
from rouge import Rouge
import time
from utils import *
from model.decodeMethod import greedy_decode
from model.dataset import PrepareData
from model.Transformer import make_model

def evaluate(data, model):
    """ 2024.07.14
    在data上用训练好的模型进行预测，打印模型翻译结果
    :param data: 数据
    :param model: 训练好的模型
    """
    bleu_scores = []
    rouge1_f1_scores = []
    rougeL_f1_scores = []
    # 初始化 ROUGE 评估工具
    rouge = Rouge()
    # 梯度清零
    with torch.no_grad():
        # 在data的英文数据长度上遍历下标
        for i in range(len(data.dev_en)):
            # 打印待翻译的英文语句
            en_sent = " ".join([data.en_index_dict[w] for w in data.dev_en[i]])
            print("\n" + en_sent)
            # 打印对应的中文语句答案
            cn_sent = " ".join([data.cn_index_dict[w] for w in data.dev_cn[i]])
            print("".join(cn_sent))
            # 将当前以单词id表示的英文语句数据转为tensor，并放入DEVICE中
            src = torch.from_numpy(np.array(data.dev_en[i])).long().to(DEVICE)
            # 增加一维
            src = src.unsqueeze(0)
            # 设置attention mask
            src_mask = (src != 0).unsqueeze(-2)
            # 用训练好的模型进行decode预测
            out = greedy_decode(model, src, src_mask, max_len=MAX_LENGTH, start_symbol=data.cn_word_dict["BOS"])
            # out = top_p_decode(model, src, src_mask, max_len=MAX_LENGTH, start_symbol=data.cn_word_dict["BOS"], top_p=0.95)
            # 初始化一个用于存放模型翻译结果语句单词的列表
            translation = []
            # 遍历翻译输出字符的下标（注意：开始符"BOS"的索引0不遍历）
            for j in range(1, out.size(1)):
                # 获取当前下标的输出字符
                sym = data.cn_index_dict[out[0, j].item()]
                # 如果输出字符不为'EOS'终止符，则添加到当前语句的翻译结果列表
                if sym != 'EOS':
                    translation.append(sym)
                # 否则终止遍历
                else:
                    break
            # 打印模型翻译输出的中文语句结果
            print("translation: %s" % " ".join(translation))
            # 计算BLEU分数
            reference = cn_sent.split()  # 参考翻译，这里假设已经是tokenized的形式
            # 去除reference里的'EOS'和'BOS'标记
            reference = reference[1:-1]
            candidate = translation  # 候选翻译
            if len(reference) == 0 or len(candidate) == 0:
                print("Empty reference or candidate, skipping BLEU and ROUGE calculation.")
                continue
            bleu = sacrebleu.sentence_bleu(" ".join(candidate), [" ".join(reference)])
            bleu_scores.append(bleu.score)
            # 打印bleu
            print(f"BLEU Score: {bleu.score} ")
            # 计算ROUGE分数
            reference_tokens = " ".join(reference)
            candidate_tokens = " ".join(candidate)
            scores = rouge.get_scores(candidate_tokens, reference_tokens, avg=True)
            rouge1_f1_scores.append(scores['rouge-1']['f'])
            rougeL_f1_scores.append(scores['rouge-l']['f'])
            # 打印ROUGE
            print(f"ROUGE-1 F1 Score: {scores['rouge-1']['f']}")
            print(f"ROUGE-L F1 Score: {scores['rouge-l']['f']}")

    # 输出整个验证集上的BLEU和ROUGE平均值
    avg_bleu = np.mean(bleu_scores)
    avg_rouge1 = np.mean(rouge1_f1_scores)
    avg_rougeL = np.mean(rougeL_f1_scores)
    print(f"\nAverage BLEU Score: {avg_bleu}")
    print(f"Average ROUGE-1 F1 Score: {avg_rouge1}")
    print(f"Average ROUGE-L F1 Score: {avg_rougeL}")

if __name__ == '__main__':
    # 数据预处理
    data = PrepareData(TRAIN_FILE, DEV_FILE)
    src_vocab = len(data.en_word_dict)
    tgt_vocab = len(data.cn_word_dict)
    print("src_vocab %d" % src_vocab)
    print("tgt_vocab %d" % tgt_vocab)
    # 初始化模型
    model = make_model(src_vocab, tgt_vocab, LAYERS, D_MODEL, D_FF, H_NUM, DROPOUT)
    # 将模型放入DEVICE中
    model.to(DEVICE)
    # 加载训练好的模型参数
    model.load_state_dict(torch.load(SAVE_FILE))
    print(">>>>>>> start evaluate")
    evaluate_start  = time.time()
    # 在验证集上进行预测
    evaluate(data, model)
    print(f"<<<<<<< finished evaluate, cost {time.time()-evaluate_start:.4f} seconds")