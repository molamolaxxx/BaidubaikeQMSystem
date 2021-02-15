import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics   # 评估模型
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 读取CSV数据，并创建数据表
    csv_data = pd.read_csv('./Result.csv', header=0)
    ds = csv_data.values[:, 1:]
    # 聚类算法K-means
    k_means = KMeans(n_clusters=4, random_state=100)
    k_means.fit(ds)

    labels = k_means.labels_
    print(k_means.cluster_centers_)  # 查看聚类中心 k_means.cluster_centers_
    print(labels)  # 查看聚类中心 k_means.cluster_centers_
    pd.plotting.scatter_matrix(csv_data)
    plt.show()

