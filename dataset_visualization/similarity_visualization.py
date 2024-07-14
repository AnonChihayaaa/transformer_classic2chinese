import pandas as pd
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
import torch
import seaborn as sns

# 去除特殊符号的函数
def remove_special_characters(text):
    return re.sub(r'[！？“”《》…，。；：、（）【】]', '', text)

# 读取数据集
data = pd.read_csv('古文-现代文语料集.csv', delimiter=',', header=None, names=['古文', '现代文'])

# 去除特殊符号
data['古文'] = data['古文'].apply(remove_special_characters)
data['现代文'] = data['现代文'].apply(remove_special_characters)

# 初始化Sentence-BERT模型
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 计算句子嵌入向量，添加进度条
ancient_embeddings = []
modern_embeddings = []

print("Encoding ancient texts:")
for text in tqdm(data['古文'].tolist(), desc="Ancient Texts"):
    ancient_embeddings.append(model.encode(text, convert_to_tensor=True))

print("Encoding modern texts:")
for text in tqdm(data['现代文'].tolist(), desc="Modern Texts"):
    modern_embeddings.append(model.encode(text, convert_to_tensor=True))

# 转换为Tensor
ancient_embeddings = torch.stack(ancient_embeddings)
modern_embeddings = torch.stack(modern_embeddings)

# 计算相似度，添加进度条
similarities = []
print("Calculating similarities:")
for ancient_emb, modern_emb in tqdm(zip(ancient_embeddings, modern_embeddings), total=len(data), desc="Calculating Similarities"):
    similarity = util.pytorch_cos_sim(ancient_emb, modern_emb).item()
    similarities.append(similarity)

# 添加相似度列到数据集中
data['相似度'] = similarities

# 输出结果
print(data[['古文', '现代文', '相似度']])

# 可视化相似度分布
plt.figure(figsize=(10, 6))
sns.histplot(data['相似度'], bins=20, kde=True, color='skyblue', edgecolor='black')
plt.xlabel('Similarity', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Similarity Distribution between Ancient and Modern Texts', fontsize=16)
plt.axvline(data['相似度'].mean(), color='red', linestyle='dashed', linewidth=1)
min_ylim, max_ylim = plt.ylim()
plt.text(data['相似度'].mean()*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(data['相似度'].mean()), color='red')
plt.grid(True, linestyle='--', alpha=0.6)
plt.style.use('seaborn-paper')
plt.show()
