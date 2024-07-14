import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 数据
length_metrics = {
    'Metric': ['Ancient Avg Length', 'Modern Avg Length', 'Ancient Min Length', 'Modern Min Length', 'Ancient Max Length', 'Modern Max Length'],
    'Value': [18.0, 28.18, 1, 2, 245, 393]
}

# 将数据转换为DataFrame
df_length = pd.DataFrame(length_metrics)

# 设置图表样式
sns.set(style="whitegrid")

# 创建子图
plt.figure(figsize=(12, 6))

# 绘制条形图
sns.barplot(x='Metric', y='Value', data=df_length, palette='Blues_d')
plt.title('Sentence Length Statistics', fontsize=16)
plt.ylabel('Value', fontsize=14)
plt.xticks(rotation=45, horizontalalignment='right')

# 标注具体数值
for p in plt.gca().patches:
    plt.gca().annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='baseline', fontsize=12, color='black', xytext=(0, 5), textcoords='offset points')

# 调整布局和字体
plt.tight_layout()
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 显示网格线
plt.grid(True, linestyle='--', alpha=0.6)

plt.show()
