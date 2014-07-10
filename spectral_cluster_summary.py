# -*- coding: utf-8 -*-
"""
#尝试调用各种第三方库里面的spectral cluster 方法
#主要尝试sklearn中cluster所包含的各种聚类的算法：
#如KMenas、k_means、k_means_、SpectralClustering、SpectralCoclustering、spectral、spectral_clustering
"""
import numpy as np
from sklearn import cluster, datasets
import kmeans


def load_data(filename):
    data = np.loadtxt(filename,dtype=np.float)

    return data


#KMeans方法
def call_KMeans(data, n_clusters):
    """class sklearn.cluster.KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001,\
      precompute_distances=True, verbose=0, random_state=None, copy_x=True, n_jobs=1)

      average complexity is given by O(k n T), were n is the number of samples and T is the number of iteration.
      该函数包含了各种参数，其中：
      n_clusters为聚类的个数，默认为8
      init为开始的k个中心点的初始方法，{'k-means++', 'random' or an ndarray}
      n_init为算法在不同的随机中心的情况下执行的次数
      max_iter为算法的最大迭代次数
      tol为阈值，表明收敛
      n_jobs为计算的任务数量，并行{-1:所有的cpus,1：没有并行,-2：只有一个cpu不参与}

     属性：
     cluster_centers_    # 聚类已有的中心点数组
     labels_             # 聚类以后的标签
     函数：
     fit(data)  Compute k-means clustering
     fit_predict(self, X)
     predict(self, X)
    """

    #datasets中包含了很多常用的数据库
    #iris是经典的花的种类的数据
    iris = datasets.load_iris()
    k_means = cluster.KMeans(n_clusters)
    #k_means.fit(iris.data)
    k_means.fit(data)

    k_clusters = k_means.cluster_centers_
    k_labels = k_means.labels_

    print "中心点坐标：\n", k_clusters
    print "类标签：\n", k_labels

    return k_labels


#k_means方法
def call_k_means(data, n_clusters):
    """
    k_means(X, n_clusters, init='k-means++', precompute_distances=True, n_init=10,
     max_iter=300, verbose=False, tol=0.0001, random_state=None, copy_x=True, n_jobs=1)

    返回：
    centroid
    label
    inertia
    """
    k_clusters, k_labels, k_dis = cluster.k_means(data, n_clusters)
    print "中心点坐标：\n", k_clusters
    print "类标签：\n", k_labels
    print "距离：\n", k_dis

    return k_labels


#SpectralClustering方法
def call_SpectralClustering(data, n_clusters):
    """
    SpectralClustering(self, n_clusters=8, eigen_solver=None, random_state=None, n_init=10, gamma=1.0, affinity='rbf',
    n_neighbors=10,k=None, eigen_tol=0.0, assign_labels='kmeans', mode=None, degree=3,
    coef0=1, kernel_params=None)

    参数：
    n_clusters为类别数
    affinity {'nearest_neighbors', 'precomputed','rbf'}
    gamma：Scaling factor of RBF, polynomial, exponential chi² and sigmoid affinity kernel。
    coef0 ：Zero coefficient for polynomial and sigmoid kernels
    n_neighbors：当选取最近邻计算相似矩阵时的邻居个数
    eigen_solver: {None, 'arpack' or 'amg'} ，计算特征值和特征向量的方法
    eigen_tol：计算拉普拉斯矩阵的特征值时选取的阈值
    assign_labels : {'kmeans', 'discretize'}，计算出特征向量后，对特征向量空间聚类的方法
    kernel_params： Parameters (keyword arguments) and values for kernel passed as callable object。

    属性：
    affinity_matrix_：相似矩阵
    labels_：类标签
    方法：
    fit
    fit_predict
    """
    iris = datasets.load_iris()
    spectral_cluster = cluster.SpectralClustering(n_clusters)
    spectral_cluster.fit(data)

    spectral_w_matrix = spectral_cluster.affinity_matrix_
    spectral_labels = spectral_cluster.labels_

    return spectral_labels


#SpectralCoclustering方法
def call_SpectralCoclustering(data, n_clusters):
    """
    Spectral Co-Clustering algorithm (Dhillon, 2001)

    __init__(self, n_clusters=3, svd_method='randomized', n_svd_vecs=None, mini_batch=False,
     init='k-means++', n_init=10, n_jobs=1, random_state=None)

     svd_method:奇异值分解 {randomized' or 'arpack'}
     n_svd_vecs：在svd计算中用到的向量数
     mini_batch:决定是否用批量的kmeans方法
     init：{'k-means++', 'random' or an ndarray}
     n_init：If mini-batch k-means is used, the best initialization is chosen and the algorithm runs once

     属性：
     rows_：shape (n_row_clusters, n_rows) 每个样本在某一个距离的列表的中关系，在为Ture,不在为False
     columns_:shape (n_column_clusters, n_columns)
     row_labels_:shape (n_rows,)
     column_labels_:shape (n_cols,)

     方法：
     fit(self, X)
    """
    spectral_cluster = cluster.SpectralCoclustering(n_clusters, mini_batch = True)
    spectral_cluster.fit(data)

    return spectral_cluster.row_labels_


if __name__ == '__main__':
    data = load_data('F:\machine learning\PythonCode\dataset\kmeansTestData.txt')
    data = kmeans.normal_data(data)
    # 聚类的数量
    n_clusters = 4
    # 描绘原数据
    kmeans.mat_plot_sample(data)
    #labels = call_k_means(data,n_clusters)
    #labels = call_KMeans(data,n_clusters)
    labels = call_SpectralClustering(data, n_clusters)
    #labels = call_SpectralCoclustering(data,n_clusters)
    # 描绘处理后的数据
    kmeans.mat_plot_cluster_sample(data, n_clusters, labels)




