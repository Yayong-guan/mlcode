#-*- coding:utf-8 -*-
__author__ = 'guanyayong'

import numpy as np
import copy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from matplotlib.font_manager import FontProperties


# 决策树，ID3算法，C4.5算法，包括特征选择、决策树的生成、决策树的修剪
#ID3算法和C4.5算法的区别在于划分特征点的选取，ID3使用的是特征值的信息增益来划分，C4.5使用的是特征的信息增益比来划分

# 全局的特征属性列表，包含初始的所有属性
original_attribute_list = list()


class DecisionTree:
    def __init__(self):
        self.attribute = None        # 节点特征属性
        self.split = None            # 划分特征属性值的分割点
        self.children = dict()       # 节点的孩子列表
        self.values = None           # 节点属性的样本值
        self.parent = None           # 节点的父节点
        self.is_leaf = False         # 节点是否为叶子节点
        self.label = None            # 叶节点所属的类别


# 计算经验熵
def calc_entropy(data):
    """
    label表示样本的类别Y
    """
    data = np.array(data)
    label = list(data[:, -1])

    m = len(data)
    label_class = np.unique(label)

    entropy = float(0.0)
    for item in label_class:
        number = label.count(item)
        entropy += (float(number) / float(m)) * np.log2(float(number) / float(m))

    entropy = 0.0 - entropy

    return entropy


# 选取最优的划分属性
def select_optimal_attribute(data, attribute_list):
    """
    :param data: 剩余的划分数据
    :param attribute_list: 剩余的特征属性列表
    :return: 返回最优的划分特征属性、该特征属性下的划分值、该特征属性的信息增益、小于划分值的样本数据、大于划分值的样本数据
    """
    data = np.array(data)
    m = len(data)

    optimal_attribute = None
    optimal_data_less_pivot = None
    optimal_data_more_pivot = None
    optimal_split = None
    children_data = dict()

    max_gain_info = None

    for attribute in attribute_list:
        col = original_attribute_list.index(attribute)
        gain_info = float(0.0)
        condition_entropy = float(0.0)
        h_entropy = float(0.0)

        data_without_label = np.array([map(float, data[i, 0:-1]) for i in range(m)])
        pivot = np.mean(data_without_label[:, col])
        data_attribute_less_pivot = data[np.nonzero(data_without_label[:, col] < pivot)[0], :]
        data_attribute_more_pivot = data[np.nonzero(data_without_label[:, col] >= pivot)[0], :]

        num_less_pivot = len(data_attribute_less_pivot)
        num_more_pivot = len(data_attribute_more_pivot)

        condition_entropy += num_less_pivot / float(m) * calc_entropy(data_attribute_less_pivot)
        condition_entropy += num_more_pivot / float(m) * calc_entropy(data_attribute_more_pivot)

        h_entropy -= num_less_pivot / float(m) * np.log2(num_less_pivot / float(m))
        h_entropy -= num_more_pivot / float(m) * np.log2(num_more_pivot / float(m))
        #print(h_entropy)
        # ID3算法获取的信息增益
        gain_info = calc_entropy(data) - condition_entropy
        # C4.5算法的获取的信息增益比
        gain_info /= h_entropy
        #print gain_info
        if max_gain_info < gain_info:
            max_gain_info = gain_info
            optimal_attribute = attribute
            optimal_split = pivot
            optimal_data_less_pivot = data_attribute_less_pivot
            optimal_data_more_pivot = data_attribute_more_pivot

    children_data['<pivot'] = optimal_data_less_pivot
    children_data['>=pivot'] = optimal_data_more_pivot
    #children_data.append(optimal_data_less_pivot)
    #children_data.append(optimal_data_more_pivot)
    #print("more")
    #print(optimal_data_more_pivot)
    #print("less")
    #print(optimal_data_less_pivot)

    return optimal_attribute, optimal_split, max_gain_info, children_data


def max_samples(labels, label_class):
    """
    :param labels: 所有样本的类别值列表
    :param label_class: 样本包含的不重复的类别列表
    :return: 返回实例数最大的类别
    """
    max_label = None
    max_label_number = int(0)

    for label in label_class:
        temp_number = labels.count(label)
        if max_label_number < temp_number:
            max_label_number = temp_number
            max_label = label

    return max_label


def create_decision_tree(data, attribute_list, root, r=0.000000001):
    """
    data： 数据集:D
    attribute_list： 特征属性列表：A
    r： 划分阈值，当信息增益小于该阈值时，停止划分
    """
    data = np.array(data)
    labels = list(data[:, -1])
    label_class = np.unique(labels)

    max_label = max_samples(labels, label_class)

    # 如果剩余的数据集都属于同一类C，表明该节点是叶子节点，子节点为空
    if len(label_class) == 1:
        root.is_leaf = True
        root.label = label_class[0]
        return root.label

    # 如果特征属性列表为空，返回节点node为叶节点，标记实例最多的类Ck为类标记
    if len(attribute_list) == 0:
        root.is_leaf = True
        root.label = max_label
        return root.label

    #计算属性列表A中的各个特征对数据集D的信息增益，选择信息增益最大的特征值Ag
    else:
        # 调用选择最优特征属性的函数，返回最优的特征属性、以及该特征的信息增益值和在该特征下的各种取值
        optimal_attribute, optimal_split, max_gain_info, \
            child_data = select_optimal_attribute(data, attribute_list)

        node_dict = {optimal_attribute: {}}

        # 信息增益大于阈值情况
        if max_gain_info > r:
            col = original_attribute_list.index(optimal_attribute)

            root.attribute = optimal_attribute
            root.split = optimal_split
            root.values = list(np.unique(data[:, col]))
            attribute_list.remove(optimal_attribute)

            for key in child_data.keys():
                node = DecisionTree()
                node.parent = root
                root.children[key] = node

                if len(child_data[key]) == 0:
                    node.is_leaf = True
                    node.label = None
                    return node.label
                else:
                    temp_attribute_list = copy.deepcopy(attribute_list)
                    node_dict[optimal_attribute][key] = \
                        create_decision_tree(child_data[key], temp_attribute_list, node)
        # 信息增益小于阈值情况
        else:
            root.is_leaf = True
            #root.label = max_label
            return root.label

    return node_dict

def load_iris_data(filename):
    all_data = np.loadtxt(filename, dtype=np.str, delimiter=',')

    attribute_list = list(all_data[0, :])
    attribute_list.pop(-1)
    labels = all_data[1:, -1]
    data = all_data[1:, :]

    nums_values_list = [len(np.unique(data[:, i])) for i in range(len(attribute_list))]

    return data, labels, attribute_list


def main():
    data, labels, attribute_list = load_iris_data('F:\machine learning\PythonCode\dataset\iris.txt')
    root = DecisionTree()
    global original_attribute_list
    original_attribute_list = copy.deepcopy(attribute_list)
    a = create_decision_tree(data, attribute_list, root)
    print(a)
    print_decision_tree(root)


decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,
    xycoords='axes fraction',
    xytext=centerPt, textcoords='axes fraction',
    va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def createPlot():

    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotNode(U'决策节点', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode(U'叶节点', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()


#层序遍历
def print_decision_tree(root):
    node_list = list()
    high = int(0)
    node_list.append(root)
    level_number = int(0)

    print' ' * high, root.attribute
    flag = len(root.children.values())

    for value in root.children.values():
        node_list.append(value)
    node_list.pop(0)
    high += 1

    while len(node_list) != 0:
        node = node_list[0]
        if node.is_leaf:
            print'     ' * high, node.label
        else:
            print '     ' * high, node.attribute
        for value in node.children.values():
            node_list.append(value)
            if flag != 0:
                level_number += 1
        node_list.pop(0)
        flag -= 1

        if flag == 0:
            high += 1
            flag = level_number
            level_number = 0

if __name__ == "__main__":
    main()
    matplotlib.rcParams['font.family'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    #createPlot()
