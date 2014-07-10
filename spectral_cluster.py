# -*- coding: utf-8 -*-

import numpy as np
from numpy.matlib import repmat
import kmeans
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, Ward
from sklearn import datasets
import bikmeans
import time


# 加载数据
def load_data(filename):
    data_set = np.loadtxt(filename, dtype=np.float)

    return data_set


def load_data_with_label(filename):
    all_data = np.loadtxt(filename, dtype=np.float, delimiter='\t')
    data = all_data[:, 0:-1]
    label = all_data[:, -1]

    return data, label


def normal_eigen(data_set):
    normal_coeff = np.sum(data_set ** 2, 1) ** 0.5
    m = len(data_set)

    for i in range(m):
        try:
            data_set[i, :] *= 1. / normal_coeff[i]
        except ZeroDivisionError:
            print("0为除数")

    return data_set


# 归一化数据
def normal_data(data_set):
    mean = np.mean(data_set, 0)
    std = np.std(data_set, 0)
    try:
        data_set = (data_set - mean) / std
    except ZeroDivisionError:
        print("0为除数")

    return data_set


# 返回数组中的第中个最小元素的下标的启动函数，不破坏原数组
def get_k_min_index(data, k):
    m = len(data)
    if m < k:
        return -1
    # 创建一个跟踪数组，其内容为原数组中元素的下标，
    # 用于记录元素的交换（即代替元素的交换）
    # 按顺序以track数组中的数据为下标访问元素
    track = list()
    for i in range(m):
        track.append(i)
    index = calc_k_min(data, track, 0, m - 1, k)

    return index


# 递归获取第K小元素
def calc_k_min(data, track, left, right, k):
    centre = data[track[right]]
    i = left
    j = right - 1
    while True:
        while data[track[i]] < centre:
            i += 1
        # 从后向前扫描时要检查下标，防止数组越界
        while j >= left and data[track[j]] > centre:
            j -= 1
        # 如果没有完成一趟交换，则交换，注意，是交换跟踪数组的值
        if i < j:
            track[i], track[j] = track[j], track[i]
        else:
            break
    # 把枢纽放在正确的位置
    track[i], track[right] = track[right], track[i]
    # 如果此时centre的位置刚好为k，则centre为第k个最小的数，
    # 返回其在真实数组中的下标，即track[i]
    if (i + 1) == k:
        return track[i]
    elif (i + 1) < k:
        # 如果此时centre的位置比k前,递归地在其右边寻找
        k_min = calc_k_min(data, track, i + 1, right, k)
    else:
        # 如果此时centre的位置比k后,递归地在其左边寻找
        k_min = calc_k_min(data, track, left, i - 1, k)

    return k_min


# 获取拉普拉斯矩阵，运用self-tuning算法
def get_lap_matrix_self_tuning(data):
    start_time = time.clock()
    sls2 = -2 * np.mat(data) * np.mat(data.T)
    sls1 = np.mat(np.sum(data ** 2, 1))
    w_matrix = sls2 + repmat(sls1, len(sls1), 1) + repmat(sls1.T, 1, len(sls1))
    w_matrix = np.array(w_matrix)

    sigma = list()
    m = len(w_matrix)
    sort_w_matrix = np.sort(w_matrix)
    for i in range(m):
        sigma.append(np.sqrt(sort_w_matrix[i, 7]))
    """for i in range(m):
        print(i)
        idx = np.argsort(w_matrix[i, :])

        idx = get_k_min_index(w_matrix[i, :].tolist(), 7)
        sigma.append(np.sqrt(w_matrix[i, idx]))
    """

    for row in range(m):
        for col in range(m):
            w_matrix[row][col] /= float(sigma[row] * sigma[col])

    w_matrix = np.exp(-np.mat(w_matrix))
    w_matrix = np.array(w_matrix)
    d_matrix = np.diag(np.sum(w_matrix, 1))
    d_matrix_square_inv = np.linalg.inv(d_matrix ** 0.5)
    dot_matrix = np.dot(d_matrix_square_inv, w_matrix)
    lap_matrix = np.dot(dot_matrix, d_matrix_square_inv)
    end_time = time.clock()

    print("calc self_tuning laplace matrix spends %f seconds" % (end_time - start_time))
    return lap_matrix


# 获取相似矩阵，运用传统的方法
def get_lap_matrix_sl(data, sigma):
    #高斯核函数
    # s(x_i, x_j) = exp(-|x_i - x_j|^2 /(2 * sigma^2))
    start_time = time.clock()
    sls2 = -2 * np.mat(data) * np.mat(data.T)
    sls1 = np.mat(np.sum(data ** 2, 1))
    w_matrix = np.exp(-(sls2 + repmat(sls1, len(sls1), 1) + repmat(sls1.T, 1, len(sls1))) / float((2 * sigma ** 2)))
    w_matrix = np.array(w_matrix)
    d_matrix = np.diag(np.sum(w_matrix, 1))
    lap_matrix = d_matrix - w_matrix
    end_time = time.clock()

    print("calc sl laplace matrix spends %f seconds" % (end_time - start_time))
    return lap_matrix


# 获取拉普拉斯矩阵，运用NJW方法
def get_lap_matrix_njw(data, sigma):
    #高斯核函数
    # s(x_i, x_j) = exp(-|x_i - x_j|^2 /(2 * sigma^2))
    start_time = time.clock()
    sls2 = -2 * np.mat(data) * np.mat(data.T)
    sls1 = np.mat(np.sum(data ** 2, 1))
    w_matrix = np.exp(-(sls2 + repmat(sls1, len(sls1), 1) + repmat(sls1.T, 1, len(sls1))) / float((2 * sigma ** 2)))
    w_matrix = np.array(w_matrix)
    d_matrix = np.diag(np.sum(w_matrix, 1))
    d_matrix_square_inv = np.linalg.inv(d_matrix ** 0.5)
    dot_matrix = np.dot(d_matrix_square_inv, w_matrix)
    lap_matrix = np.dot(dot_matrix, d_matrix_square_inv)
    end_time = time.clock()

    print("calc njw laplace matrix spends %f seconds" % (end_time - start_time))
    return lap_matrix


# 计算矩阵的特征值和特征向量
def calc_eigenvectors(lap_matrix):
    print("Calculating the eigenvectors...")
    start_time = time.clock()
    eigenvalues, eigenvectors = np.linalg.eig(lap_matrix)
    end_time = time.clock()
    # 从小到大排序
    idx = eigenvalues.argsort()
    #eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    print("calc njw the eigenvectors spends %f seconds" % (end_time - start_time))
    return eigenvectors


# 谱聚类函数实现
def spectral_cluster(data, n_clusters, method='sl'):
    # 获取拉普拉斯矩阵
    if method == 'NJW':
        lap_matrix = get_lap_matrix_njw(data, 0.1)
        eigenvalues, eigenvectors = np.linalg.eig(lap_matrix)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

    elif method == 'self-tuning':
        lap_matrix = get_lap_matrix_self_tuning(data)
        eigenvalues, eigenvectors = np.linalg.eig(lap_matrix)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

    else:
        lap_matrix = get_lap_matrix_sl(data, 0.1)
        eigenvalues, eigenvectors = np.linalg.eig(lap_matrix)
        idx = eigenvalues.argsort()
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

    #print(eigenvalues)
    # 获取前n_clusters个特征向量
    x_matrix = eigenvectors[:, 0:n_clusters]
    # 归一化特征向量矩阵
    y_matrix = normal_eigen(x_matrix)

    # 调用自己写的k_means函数
    """
    k_dist_dic, k_centers_dic, cluster_group = kmeans.k_means(y_matrix, n_clusters)
    mat_plot_cluster_sample(data, cluster_group, method)
    """
    # 调用自己写的bi_k_means函数
    """center_list, cluster_assign = bikmeans.exe_bi_k_means(y_matrix, n_clusters)
    labels = cluster_assign[:, 0]
    mat_plot_cluster_sample(data, labels. method)

    # 调用sklearn中的KMeans函数，效果比自己写的强了好多
    k_means = KMeans(n_clusters)
    k_means.fit(y_matrix)
    #k_centers = k_means.cluster_centers_
    #mat_plot_cluster_sample(data, k_means.labels_, method)
    """
    # 调用sklearn中的hierarchical 聚类方法进行聚类
    hie_cluster = Ward(n_clusters)
    hie_cluster.fit(y_matrix)
    mat_plot_cluster_sample(data, hie_cluster.labels_, method)


# plot聚类后的样本数据
def mat_plot_cluster_sample(data_set, cluster_group, method):
    fig = plt.figure()
    ax = fig.gca()
    color_samples = ['ro', 'bo', 'go', 'yo', 'wo']

    n_cluster = np.unique(cluster_group)
    for i in n_cluster:
        group_data = data_set[np.nonzero(cluster_group[:] == i)[0], :]
        ax.plot(group_data[:, 0], group_data[:, 1], color_samples[int(i)])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.title(method + " spectral cluster algorithms")
    plt.draw()


def main():
    #data, label = datasets.load_iris().data, datasets.load_iris().traget
    #data = load_data('F:\machine learning\PythonCode\dataset\cluster\spectral_cluster_data.txt')
    data, label = load_data_with_label('F:\machine learning\PythonCode\dataset\cluster\Spiral.txt')
    data = normal_data(data)
    #显示原来的数据分布
    plt.ion()
    #kmeans.mat_plot_sample(data)
    mat_plot_cluster_sample(data, label, 'original sample')
    # k-means聚类，对比实验
    k_dist1, k_centers1, cluster_group = kmeans.k_means(data, 3)
    kmeans.mat_plot_k_means_sample(k_dist1, k_centers1)
    # 谱聚类，第三个参数为聚类是采用的算法，有'NJW'、'self-tuning'、'sl',默认为'sl'
    spectral_cluster(data, 3)
    spectral_cluster(data, 3, 'NJW')
    spectral_cluster(data, 3, 'self-tuning')
    plt.ioff()
    plt.show()

# 主函数
if __name__ == '__main__':
    main()
