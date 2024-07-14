import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import STOPWORDS
import jieba
import re
from scipy.spatial.distance import cosine
from transformers import BertTokenizer, BertModel

# 去除特殊符号的函数
def remove_special_characters(text):
    return re.sub(r'[！？“”《》…，。；：、（）【】]', '', text)

# 读取数据集
data = pd.read_csv('古文-现代文语料集.csv', delimiter=',', header=None, names=['古文', '现代文'])

# 去除特殊符号
data['古文'] = data['古文'].apply(remove_special_characters)
data['现代文'] = data['现代文'].apply(remove_special_characters)

# 计算句对数量
num_pairs = len(data)

# 计算句子长度（字符数和词数）
data['古文长度'] = data['古文'].apply(len)
data['现代文长度'] = data['现代文'].apply(len)
data['古文词数'] = data['古文'].apply(lambda x: len(jieba.lcut(x)))
data['现代文词数'] = data['现代文'].apply(lambda x: len(jieba.lcut(x)))

# 句子长度统计
ancient_length_stats = data['古文长度'].describe()
modern_length_stats = data['现代文长度'].describe()

# 计算词汇大小
ancient_vocab = set()
modern_vocab = set()

for text in data['古文']:
    ancient_vocab.update(jieba.lcut(text))

for text in data['现代文']:
    modern_vocab.update(jieba.lcut(text))

ancient_vocab_size = len(ancient_vocab)
modern_vocab_size = len(modern_vocab)

# 计算词频分布
ancient_word_freq = Counter()
modern_word_freq = Counter()

for text in data['古文']:
    ancient_word_freq.update(jieba.lcut(text))

for text in data['现代文']:
    modern_word_freq.update(jieba.lcut(text))



# 绘制词频分布图
def plot_word_freq(word_freq, title, top_n=30):
    common_words = word_freq.most_common(top_n)
    words, counts = zip(*common_words)
    plt.figure(figsize=(10, 8))
    plt.barh(words, counts, color='skyblue')
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.style.use('seaborn-paper')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.show()

plot_word_freq(ancient_word_freq, 'Ancient Text Word Frequency Distribution')
plot_word_freq(modern_word_freq, 'Modern Text Word Frequency Distribution')

# 输出统计结果
print(f'句对数量: {num_pairs}')
print(f'古文平均长度: {ancient_length_stats["mean"]}, 最短长度: {ancient_length_stats["min"]}, 最长长度: {ancient_length_stats["max"]}')
print(f'现代文平均长度: {modern_length_stats["mean"]}, 最短长度: {modern_length_stats["min"]}, 最长长度: {modern_length_stats["max"]}')
print(f'古文词汇大小: {ancient_vocab_size}')
print(f'现代文词汇大小: {modern_vocab_size}')
print(f'古文常见词: {ancient_word_freq.most_common(10)}')
print(f'现代文常见词: {modern_word_freq.most_common(10)}')
