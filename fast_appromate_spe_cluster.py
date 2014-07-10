# *-* coding: utf-8 *-*
"""
fast approximate spectral cluster
_author__ = 'guanyayong'
"""

from sklearn.cluster import KMeans, SpectralClustering
from sklearn import datasets
import numpy as np
import itertools
import time

iris = datasets.load_iris()
iris_data = iris.data
iris_label = iris.target


def fast_app_spe_cluster(data, label, k, n_cluster):
    #k-means get the representative points(centers points)
    start_time = time.clock()
    k_means = KMeans(n_clusters=k)
    k_means.fit(data)
    y_centers = k_means.cluster_centers_
    # get the correspondence table
    x_to_centers_table = list()
    m = len(data)
    for i in range(m):
        min_distance = np.inf
        min_index = None
        for j in range(k):
            i_j_dis = np.sum((data[i, :] - y_centers[j, :]) ** 2)
            if min_distance > i_j_dis:
                min_index = j
                min_distance = i_j_dis
        x_to_centers_table.append(min_index)
    # spectral cluster
    spe_cluster = SpectralClustering(n_clusters=n_cluster)
    spe_cluster.fit(y_centers)
    spe_label = spe_cluster.labels_
    # get m-way cluster membership
    x_label = list()
    for i in range(m):
        x_label.append(spe_label[x_to_centers_table[i]])
    spend_time = time.clock() - start_time
    print("spend time is %f seconds" % spend_time)
    return x_label


def calc_accuracy(real_label, estimate_label):
    assert len(real_label) == len(estimate_label)

    m = len(real_label)
    correct = int(0)
    est_unique_label = np.unique(est_label)
    permutation = list(itertools.permutations(est_unique_label))
    max_corrcet = -np.inf
    permutation_label = np.zeros(m)
    for i in range(1, len(permutation)):
        correct = int(0)
        for j in range(m):
            for k in range(len(est_unique_label)):
                if est_label[j] == permutation[0][k]:
                    permutation_label[j] = permutation[i][k]
                    break
            if real_label[j] == permutation_label[j]:
                correct += 1
        if max_corrcet < correct:
            max_corrcet = correct

    correct_rate = max_corrcet / float(m) * 100
    print("correct rate is %f" % correct_rate)


if __name__ == '__main__':
    est_label = fast_app_spe_cluster(iris_data, iris_label, 30, 3)
    calc_accuracy(iris_label, est_label)