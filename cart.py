#-*- coding: utf-8 -*-
"""
CART 分类与回归树
Created by guanyayong
2014/6/26
"""
import numpy as np
import copy
import six
from warnings import warn
import pydot
import random
#import sklearn
from sklearn import datasets


class CART:
    ID = int(0)
    origin_features_list = list()

    def __init__(self, data, features_list):
        self.feature = None
        self.parent = None
        self.children = dict()
        self.criterion = None
        self.impurity = 0.0000
        self.threshold = None
        self.is_leaf = False
        self.label = None
        self.data = data
        self.features_list = features_list
        self.node_id = None

    def calc_least_squares(self):
        min_least_squares = np.inf
        optimal_feature = None
        optimal_feature_threshold = None

        for feature in self.features_list:
            index = CART.origin_features_list.index(feature)
            feature_values = np.unique(self.data[:, index]).tolist()

            for value in feature_values:
                sum_less_least_squares = \
                    sum((self.data[np.nonzero(self.data[:, index] <= value)[0], -1] - 0) ** 2)

                sum_more_least_squares = \
                    sum((self.data[np.nonzero(self.data[:, index] > value)[0], -1] - 1) ** 2)

                sum_value_least_squares = sum_less_least_squares + sum_more_least_squares

                if sum_more_least_squares < min_least_squares:
                    min_least_squares = sum_value_least_squares
                    optimal_feature = feature
                    optimal_feature_threshold = value

        self.feature = optimal_feature
        self.threshold = optimal_feature_threshold
        self.criterion = "least_squares"
        self.impurity = min_least_squares

    def calc_gini(self, data1, data2):

        label1_values = data1[:, -1].tolist()
        label2_values = data2[:, -1].tolist()

        label1_unique = np.unique(label1_values).tolist()
        label2_unique = np.unique(label2_values).tolist()

        num_label1 = len(label1_values)
        num_label2 = len(label2_values)

        ratio1 = len(data1) / float(len(self.data))
        ratio2 = 1 - ratio1

        label1_ratio1 = float(0.0)
        label2_ratio2 = float(0.0)

        for item in label1_unique:
            temp_num = int(0)
            for label in label1_values:
                if item == label:
                    temp_num += 1
            label1_ratio1 += temp_num * temp_num

        label1_ratio1 /= num_label1 * num_label1

        for item in label2_unique:
            temp_num = int(0)
            for label in label2_values:
                if item == label:
                    temp_num += 1
            label2_ratio2 += temp_num * temp_num

        label2_ratio2 /= num_label2 * num_label2

        gini = ratio1 * (1 - label1_ratio1) + ratio2 * (1 - label2_ratio2)

        return gini

    def select_optimal_feature(self):
        min_gini = np.inf
        optimal_feature = None
        optimal_threshold = None
        children_data = list()
        optimal_data_less = None
        optimal_data_more = None

        for feature in self.features_list:
            index = CART.origin_features_list.index(feature)
            values = np.unique(self.data[:, index]).tolist()

            values.sort()
            for i in range(0, len(values) - 1):
                value = (values[i] + values[i + 1]) / 2.0
                data_less_value = self.data[np.nonzero(self.data[:, index] <= value)[0], :]
                data_more_value = self.data[np.nonzero(self.data[:, index] > value)[0], :]

                gini_value = self.calc_gini(data_less_value, data_more_value)

                if gini_value < min_gini:
                    min_gini = gini_value
                    optimal_feature = feature
                    optimal_threshold = value
                    optimal_data_less = data_less_value
                    optimal_data_more = data_more_value

        self.feature = optimal_feature
        self.threshold = optimal_threshold
        self.criterion = "gini"
        self.impurity = min_gini
        children_data.append(optimal_data_less)
        children_data.append(optimal_data_more)

        return children_data

    def get_most_label(self, labels, labels_unique):
        max_label = -np.inf
        max_label_number = int(0)

        for label in labels_unique:
            temp_number = labels.count(label)
            if max_label_number < temp_number:
                max_label_number = temp_number
                max_label = label

        return max_label

    def create_CART(self):
        labels = self.data[:, -1].tolist()
        labels_unique = np.unique(labels)

        most_label = self.get_most_label(labels, labels_unique)

        if len(labels_unique) == 1:
            self.is_leaf = True
            self.label = labels[0]
            return self

        elif len(self.features_list) == 0:
            self.is_leaf = True
            self.label = most_label
            return self

        else:
            children_data = self.select_optimal_feature()
            temp_features_list = copy.deepcopy(self.features_list)
            temp_features_list.remove(self.feature)

            m = len(children_data)
            node_dict = {self: {}}

            for i in range(m):
                CART.ID += 1
                child_node = CART(children_data[i], temp_features_list)
                self.children[i] = child_node
                child_node.parent = self
                child_node.node_id = CART.ID

                if len(child_node.data) == 0:
                    child_node.is_leaf = True
                    child_node.label = most_label
                    return child_node
                else:
                    node_dict[self][i] = child_node.create_CART()

        return node_dict


def export_graphviz(root, out_file="tree.dot", feature_names=None,
                    max_depth=None, close=None):

    if close is not None:
        warn("The close parameter is deprecated as of version 0.14 "
             "and will be removed in 0.16.", DeprecationWarning)

    def node_to_str(tree_node, criterion):
        if not isinstance(criterion, six.string_types):
            criterion = "impurity"

        if tree_node.is_leaf:
            return "%s = %.4f\\nsamples = %s\\nlabel = %s" \
                   % (criterion,
                      tree_node.impurity,
                      len(tree_node.data),
                      tree_node.label)
        else:
            if feature_names is not None:
                feature = feature_names[tree_node.feature]
            else:
                feature = "X[%s]" % tree_node.feature

            return "%s <= %.4f\\n%s = %s\\nsamples = %s" \
                   % (feature,
                      tree_node.threshold,
                      criterion,
                      tree_node.impurity,
                      len(tree_node.data))

    def recurse(tree_node, node_id, criterion, parent=None, depth=0):

        # Add node with description
        if max_depth is None or depth <= max_depth:
            out_file.write('%d [label="%s", shape="box"] ;\n' %
                           (node_id, node_to_str(tree_node, criterion)))

            if tree_node.parent is not None:
                # Add edge to parent
                out_file.write('%d -> %d ;\n' % (parent, node_id))

            if not tree_node.is_leaf:
                for i in range(len(tree_node.children)):
                    child_node = tree_node.children[i]
                    #parent = copy.deepcopy(node_id)
                    parent = node_id
                    recurse(child_node, child_node.node_id, criterion, parent, depth=depth + 1)
        else:
            out_file.write('%d [label="(...)", shape="box"] ;\n' % node_id)

            if parent is not None:
                # Add edge to parent
                out_file.write('%d -> %d ;\n' % (parent, node_id))

    own_file = False
    if isinstance(out_file, six.string_types):
        if six.PY3:
            out_file = open(out_file, "w", encoding="utf-8")
        else:
            out_file = open(out_file, "wb")
        own_file = True

    out_file.write("digraph Tree {\n")
    if not isinstance(root, CART):
        recurse(root, 0, criterion="impurity")
    else:
        recurse(root, 0, criterion=root.criterion)
    out_file.write("}")

    if own_file:
        out_file.close()


def save_cart_to_pdf(root, filename):
    with open(filename, 'w+') as f:
        export_graphviz(root, out_file=f)

    iris_file = open(filename, 'r')
    dot_data = iris_file.read()
    graph = pydot.graph_from_dot_data(dot_data)
    graph.write_pdf("iris.pdf")


def classify_single(branch, data):
    if type(branch).__name__ != 'dict':
        return branch.label

    node = branch.keys()[0]
    col = CART.origin_features_list.index(node.feature)
    next_branch = branch[node]
    if data[col] <= node.threshold:
        next_branch = next_branch[0]
    else:
        next_branch = next_branch[1]

    return classify_single(next_branch, data)


def classify1(cart, test_data, test_label):
    num_data = len(test_data)
    num_label = len(test_label)

    assert num_data == num_label
    correct_num = int(0)
    for i in range(num_data):
        if test_label[i] == \
                classify_single(cart, test_data[i, :].tolist()):
            correct_num += 1

    correct_rate = correct_num / float(num_label)

    print("test result:")
    print("total samples number: %d" % num_label)
    print("correct samples number: %d" % correct_num)
    print("the correct rate is %4.2f%s" % (correct_rate * 100, '%'))


def classify(cart, test_data, test_label):

    num_data = len(test_data)
    num_label = len(test_label)

    assert num_data == num_label
    correct_num = int(0)
    for i in range(num_data):
        tree_node = cart.keys()[0]
        while not tree_node.is_leaf:
            col = CART.origin_features_list.index(tree_node.feature)
            if test_data[i, col] <= tree_node.threshold:
                tree_node = tree_node.children[0]
            else:
                tree_node = tree_node.children[1]

        if test_label[i] == tree_node.label:
            correct_num += 1

    correct_rate = correct_num / float(num_label)

    print("test result:")
    print("total samples number: %d" % num_label)
    print("correct samples number: %d" % correct_num)
    print("the correct rate is %4.2f%s" % (correct_rate * 100, '%'))


def load_data(filename):
    all_data = np.loadtxt(filename, dtype=np.float, delimiter=',')
    features_list = list(all_data[0, 0:-1])
    data = all_data[1:, :]

    return data, features_list


def main():
    """
    iris = datasets.load_iris()
    data_sets = iris.data
    features_list = iris.feature_names
    """
    data_sets, features_list = load_data('F:\machine learning\pythoncode\dataset\iris_data.txt')
    # 从数据集中随机选取一部分分别作为训练集和测试集
    train_data = np.array(random.sample(data_sets.tolist(), 100))
    test_data_sets = np.array(random.sample(data_sets.tolist(), 50))
    test_data = test_data_sets[:, :-1]
    test_label = test_data_sets[:, -1].tolist()
    # 训练CART树
    CART.origin_features_list = copy.deepcopy(features_list)
    root = CART(train_data, features_list)
    root.node_id = CART.ID
    # 构建CART，并保存树的节点关系在node_dict字典中
    node_dict = root.create_CART()
    # 将CART树结构图保存为pdf文件
    save_cart_to_pdf(root, 'iris.dot')
    # 测试数据的正确率
    classify1(node_dict, test_data, test_label)


if __name__ == '__main__':

    main()