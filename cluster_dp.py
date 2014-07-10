# *-* coding: utf-8 *-*
"""
learning from science paper:
Clustering by fast search and find of density peaks
coding by guanyayong
2014/7/2
"""

import numpy as np
from numpy.matlib import repmat
import matplotlib
import matplotlib.pyplot as plt
import time


class Cluster:
    dc = float(0.0)
    samples_list = list()
    distances = list()
    w_matrix = None
    n_samples = int(0)

    def __init__(self):
        self.data = None
        self.rho = float(0.0)
        self.delta = float(0.0)
        self.cluster = None

    @classmethod
    def get_dc(cls, percent):

        m = len(cls.samples_list)

        average_neighbors = percent * m

    @classmethod
    def get_local_density(cls, percent):
        m = cls.n_samples
        neighbors = int(round(percent * m))
        sort_w_matrix = np.sort(cls.w_matrix)
        neighbors_matrix = sort_w_matrix[:, 1:neighbors + 1]
        mean_dis_to_neigh = -np.sum(neighbors_matrix, axis=1) / float(neighbors)

        index_array = np.zeros(m, dtype=int)
        for k in range(m):
            cls.samples_list[k].rho = mean_dis_to_neigh[k]
            index_array[k] = k

        index_list = np.argsort(mean_dis_to_neigh).tolist()

        for i in range(m):
            index = index_list.index(i)
            if index == m - 1 and i != m - 1:
                cls.samples_list[i].delta = sort_w_matrix[i, -1]
            elif index == m - 1 and i == m - 1:
                cls.samples_list[i].delta = sort_w_matrix[i, -2]
            else:
                min_distance = np.inf
                for indices in range(index + 1, m):
                    j = index_list[indices]
                    if min_distance > cls.w_matrix[i][j]:
                        min_distance = cls.w_matrix[i][j]
                cls.samples_list[i].delta = min_distance
        """
        flag = False
        for i in range(m):
            min_distance = np.inf
            for j in range(m):
                if cls.samples_list[i].rho < cls.samples_list[j].rho:
                    flag = True
                    if min_distance > cls.w_matrix[i][j]:
                        min_distance = cls.w_matrix[i][j]
            if flag:
                cls.samples_list[i].delta = min_distance
            else:
                if i != m - 1:
                    cls.samples_list[i].delta = sort_w_matrix[i, -1]
                else:
                    cls.samples_list[i].delta = sort_w_matrix[i, -2]
        """

    @classmethod
    def get_rho_and_delta(cls, percent):
        m = cls.n_samples
        print('average percentage of neighbours (hard coded): %5.6f' % percent)
        position = int(round(m * percent / 100.))
        cls.distances.sort()
        dc = cls.distances[position]
        print('Computing Rho with gaussian kernel of radius: %12.6f' % dc)

        #Gaussian kernel
        for i in range(m - 1):
            for j in range(i + 1, m):
                cls.samples_list[i].rho += np.exp(-(cls.w_matrix[i][j] / dc) ** 2 / 2)
                cls.samples_list[j].rho += np.exp(-(cls.w_matrix[i][j] / dc) ** 2 / 2)

        """
        "Cut off kernel"
        for i in range(m - 1):
            for j in range(i + 1, m):
                if cls.w_matrix[i][j] < dc:
                    cls.samples_list[i].rho += 1.
                    cls.samples_list[j].rho += 1.
        """
        max_dis = np.max(cls.distances)
        # 定义局部变量
        rho = np.zeros(m, dtype=float)
        delta = np.zeros(m, dtype=float)
        nneigh = np.zeros(m, dtype=int)

        for i in range(m):
            rho[i] = cls.samples_list[i].rho
        # 对rho进行从小到大的排序
        """
        index_rho = np.argsort(rho).tolist()
        sort_w_matrix = np.sort(cls.w_matrix)

        for i in range(m):
            index = index_rho.index(i)
            if index == m - 1 and i != m - 1:
                cls.samples_list[i].delta = sort_w_matrix[i, -1]
            elif index == m - 1 and i == m - 1:
                cls.samples_list[i].delta = sort_w_matrix[i, -2]
            else:
                min_distance = np.inf
                for indices in range(index + 1, m):
                    j = index_rho[indices]
                    if min_distance > cls.w_matrix[i][j]:
                        min_distance = cls.w_matrix[i][j]
                        nneigh[i] = j
                cls.samples_list[i].delta = min_distance
        """
        index_rho = rho.argsort()[::-1]
        delta[index_rho[0]] = -1
        nneigh[0] = 0
        for ii in range(2, m):
            delta[index_rho[ii]] = max_dis
            for jj in range(0, ii):
                if cls.w_matrix[index_rho[ii], index_rho[jj]] < delta[index_rho[ii]]:
                    delta[index_rho[ii]] = cls.w_matrix[index_rho[ii], index_rho[jj]]
                    nneigh[index_rho[ii]] = index_rho[jj]

        delta[index_rho[0]] = np.max(delta)

        for i in range(m):
            cls.samples_list[i].delta = delta[i]

        print('Generated file:DECISION GRAPH')
        print('column 1:Density')
        print('column 2:Delta')

        outfile = open('DECISION_GRAPH.txt', 'w+')
        for i in range(m):
            string = str(cls.samples_list[i].rho) + '\t' + str(cls.samples_list[i].delta) + '\n'
            outfile.write(string)

    @classmethod
    def get_centers(cls):
        m = cls.n_samples
        gamma = np.zeros(m, dtype=float)

        for i in range(m):
            gamma[i] = cls.samples_list[i].rho * cls.samples_list[i].delta


def load_data(filename):
    data = np.loadtxt(filename, dtype=np.float, delimiter='\t')
    label = data[:, -1]
    data = data[:, 0:-1]
    clusters = np.unique(label)

    return data, label, clusters


def normal_data(data):
    mean = np.mean(data, 0)
    std = np.std(data, 0)
    try:
        data = (data - mean) / std
    except ZeroDivisionError:
        print("0为除数")

    return data


def samples_to_cluster_object(data, label):

    m = len(label)
    assert len(data) == len(label)
    Cluster.n_samples = m
    for i in range(m):
        cluster_object = Cluster()
        cluster_object.data = data[i, :]
        cluster_object.cluster = label[i]
        Cluster.samples_list.append(cluster_object)

    start_time = time.clock()
    sls2 = -2 * np.mat(data) * np.mat(data.T)
    sls1 = np.mat(np.sum(data ** 2, 1))
    Cluster.w_matrix = sls2 + repmat(sls1, len(sls1), 1) + repmat(sls1.T, 1, len(sls1))
    Cluster.w_matrix = np.array(Cluster.w_matrix) ** 0.5
    end_time = time.clock()
    print("calc distance matrix' time is %f seconds" % (end_time - start_time))

    outfile = open("distance.txt", "w+")
    print("writing distance to file ...")
    for i in range(m - 1):
        for j in range(i + 1, m):
            Cluster.distances.append(Cluster.w_matrix[i, j])
            str_d = str(i + 1) + "\t" + str(j + 1) + "\t" + str(Cluster.w_matrix[i, j]) + "\n"
            outfile.write(str_d)
    print("write done")

    del cluster_object


def mat_plot_samples(samples_list, clusters):

    ax = plt.subplot(121)

    colors = ['ro', 'bo', 'go', 'yo', 'ko', 'r.', 'b.', 'g.', 'y.', 'k.', 'r1', 'b1', 'g1', 'y1', 'k1',
              'r*', 'b*', 'g*', 'y*', 'k*', 'r^', 'b^', 'g^', 'y^', 'k^', 'rp', 'bp', 'gp', 'yp', 'kp', 'r+']

    for i in range(Cluster.n_samples):
        index = int(samples_list[i].cluster) - 1
        ax.plot(samples_list[i].data[0], samples_list[i].data[1], colors[index])
        plt.text(samples_list[i].data[0], samples_list[i].data[1], str(i))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.title("Sample Distribution")
    plt.grid()
    plt.draw()


def mat_plot_decision_graph(samples_list):

    ax = plt.subplot(122)

    colors = ['ko']
    for i in range(Cluster.n_samples):
        ax.plot(samples_list[i].rho, samples_list[i].delta, colors[0])
        plt.text(samples_list[i].rho, samples_list[i].delta, str(i))

    ax.set_xlabel('rho')
    ax.set_ylabel('delta')
    plt.title("Decision Graph")
    plt.grid()
    plt.draw()


def main():
    data, label, clusters = load_data("F:\machine learning\pythoncode\dataset\cluster\Pathbased.txt")
    data = normal_data(data)
    samples_to_cluster_object(data, label)
    plt.figure(1)
    plt.ion()
    mat_plot_samples(Cluster.samples_list, clusters)
    #Cluster.get_local_density(0.01)
    Cluster.get_rho_and_delta(1)
    mat_plot_decision_graph(Cluster.samples_list)
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()