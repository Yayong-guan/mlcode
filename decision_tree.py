#-*- coding:utf-8 -*-
__author__ = 'guanyayong'

import numpy as np
import  copy
import matplotlib.pyplot as plt


# 决策树，ID3算法，包括特征选择、决策树的生成、决策树的修剪

class DecisionTree:
    def __init__(self):
        self.attribute = None      # 节点特征属性
        self.children = dict()     # 节点的孩子列表
        self.values = None         # 节点属性的样本值
        self.parent = None         # 节点的父节点
        self.is_leaf = False       # 节点是否为叶子节点


# 计算经验熵
def calc_entropy(data):
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
def select_optimal_attribute(data, original_attribute_list, attribute_list):
    data = np.array(data)
    m = len(data)

    max_attribute = None
    max_gain_info = float(0.0)

    #print attribute_list
    for attribute in attribute_list:
        col = original_attribute_list.index(attribute)
        gain_info = float(0.0)
        condition_entropy = float(0.0)
        h_entropy = float(0.0)
        attribute_values = list(np.unique(data[:, col]))

        attribute_dict = dict()
        for item in attribute_values:
            attribute_dict[item] = list()

        for value in attribute_values:
            for row in range(m):
                if data[row, col] == value:
                    attribute_dict[value].append(list(data[row, :]))

            condition_entropy += (float(len(attribute_dict[value])) / float(m))\
                                 * calc_entropy(attribute_dict[value])

            h_entropy -= (float(len(attribute_dict[value])) / float(m)) *\
                         np.log2((float(len(attribute_dict[value])) / float(m)))

        gain_info = calc_entropy(data) - condition_entropy
        # 信息增益比
        gain_info = gain_info / h_entropy
        #print gain_info

        if max_gain_info < gain_info:
            max_gain_info = gain_info
            max_attribute = attribute
            max_values = attribute_values

    return max_attribute, max_gain_info, max_values


def load_data_1(filename):

    all_data = np.loadtxt(filename, dtype=np.str)

    attribute_list = list(all_data[0, 1:-1])
    data = all_data[1:, 1:]

    return data, attribute_list


def load_data(filename):
    all_data = np.loadtxt(filename, dtype=np.str, delimiter=',')

    attribute_list = list(all_data[0, :])
    #attribute_list.pop(0)
    attribute_list.pop(-1)
    data = all_data[1:, 0:]

    return data, attribute_list


def max_samples(labels, label_class):
    # 选出数据集中实例数最多的类别
    max_label = None
    max_label_number = int(0)
    for label in label_class:
        temp_number = labels.count(label)
        if max_label_number < temp_number:
            max_label_number = temp_number
            max_label = label

    return max_label


def create_decision_tree(data, original_attribute_list, attribute_list, root, r=0.001):
    """
    data为数据集:D
    attribute_list为特征属性列表：A
    r为阈值
    """
    data = np.array(data)
    # 获取该数据集中的所有类别
    labels = list(data[:, -1])
    label_class = np.unique(labels)
    # 选出样本数量最多的类别
    max_label = max_samples(labels, label_class)

    # 如果剩余的数据集都属于同一类C，表明该节点是叶子节点，子节点为空
    if len(label_class) == 1:
        # 返回节点node为叶节点，类C为标记
        root.attribute = label_class[0]
        root.is_leaf = True
        #print 'leaf:',root.attribute
        #print 'parent:', root.parent.attribute
        return

    # 如果特征属性列表为空，返回节点node为叶节点，标记实例最多的类Ck为类标记
    if len(attribute_list) == 0:
        root.is_leaf = True
        root.attribute = max_label

    #计算属性列表A中的各个特征对数据集D的信息增益，选择信息增益最大的特征值Ag
    else:
        # 调用选择最优特征属性的函数，返回最优的特征属性、以及该特征的信息增益值和在该特征下的各种取值
        optimal_attribute, max_gain_info, attribute_values \
            = select_optimal_attribute(data, original_attribute_list, attribute_list)
        #print 'attribute：\n', optimal_attribute

        # 将Ag标记节点node，并将划分的特征属性Ag从属性列表attribute中删除
        root.attribute = optimal_attribute    # 给该节点赋值属性
        root.values = attribute_values        # 赋值样本中该属性对应的所有值
        index = original_attribute_list.index(optimal_attribute)
        attribute_list.remove(optimal_attribute)
        # 初始化该节点的子节点
        # 按Ag中的每个取值ai将数据集D分割为若干非空的子集，递归调用
        for value in attribute_values:
            child_data = data[np.nonzero(data[:, index] == value)[0]]
            node = DecisionTree()
            node.parent = root
            root.children[value] = node

            if len(child_data) == 0:
                node.attribute = max_label
                return
            else:
                create_decision_tree(child_data, original_attribute_list, attribute_list, node)


def main():
    data, attribute_list = load_data_1('F:\machine learning\PythonCode\dataset\decisiontree.txt')
    #data, attribute_list = load_data('F:\machine learning\PythonCode\dataset\iris.txt')
    root = DecisionTree()
    original_attribute_list = copy.deepcopy(attribute_list)
    create_decision_tree(data, original_attribute_list, attribute_list, root)
    print_decision_tree(root)


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
        print'     ' * high, node.attribute
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

if __name__ == '__main__':
    main()

