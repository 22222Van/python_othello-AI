import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict


def visualize_defaultdict(file_path):
    # 从文件中加载defaultdict
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    if not isinstance(data, defaultdict):
        raise ValueError("The file does not contain a defaultdict.")

    bias = data['bias']
    del data['bias']

    # 创建8x8的矩阵用于热力图
    heatmap_data = np.zeros((8, 8))
    for (x, y), value in data.items():
        heatmap_data[x, y] = value

    # 绘制热力图
    plt.figure(figsize=(6, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f",
                cmap="YlGnBu", cbar=True, square=True)
    plt.title(f"Heatmap, bias={bias:.2f}")
    plt.xlabel("y")
    plt.ylabel("x")
    plt.show()


# 调用函数，替换为你的文件路径
file_path = "saves/q-agent-naive-oppo-cnn.pkl"
visualize_defaultdict(file_path)
