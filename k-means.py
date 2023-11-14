import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 生成示例数据，这里假设有三个不同尺寸的目标
data = np.array([[10, 10], [20, 20], [30, 30], [100, 100], [120, 120], [150, 150]])

# 初始化K均值模型，假设要分成3个簇
kmeans = KMeans(n_clusters=3)
kmeans.fit(data)

# 获取簇中心
centroids = kmeans.cluster_centers_

# 输出簇中心，这里表示三个不同尺寸的目标
print("聚类中心（目标尺寸）:\n", centroids)

# 绘制原始数据和簇中心
plt.scatter(data[:, 0], data[:, 1], c='blue', label='原始数据')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='聚类中心')
plt.legend()
plt.xlabel('宽度')
plt.ylabel('高度')
plt.title('K均值聚类示例')
plt.show()
