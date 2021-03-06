import random

import DecisionTree.tree as tree
import numpy as np


# Returns a decision tree constructed using the id3 algorithm
# with values as the attributes of the data set
# and labels as the labels of the data set. These two
# arrays should be the same length. values should be an array
# of arrays, each representing the attributes of an example,
# with the attributes in the same order for each example.
#
# if an attribute is to be treated as a numeric value,
# it should be passed in as a numeric type instead of as a string
#
# the gain_function parameter must be one of "gini", "maj error", or "info gain"
#
# If one would rather not specify a max_depth, set it equal to -1.
# attr_subset_size is used by random forests, so on each node
# a subset of this size of the attributes is used. -1 signifies no such variable.
#
# setting the all_attributes parameter should be used if the values set does not
# contain all the possible attribute values (i.e., if we are training with a subset of the data)
def id3(values, labels, gain_function, max_depth=-1, attr_subset_size=-1, all_attributes=None):
    if gain_function != "gini" and gain_function != "maj error" and gain_function != "info gain":
        raise Exception("Gain function must be equal to \"gini\", \"maj error\", or \"info gain\"")

    if all_attributes is None:
        # keeps track of all of the attribute values for each attribute
        # so the tree formed is "complete".
        all_attributes = []
        for j in range(len(values[0])):
            all_attributes.append(set())

        for j in range(len(values[0])):
            if type(values[0][j]) != str:
                all_attributes[j].add("less")
                all_attributes[j].add("more")
                continue

            for i in range(len(values)):
                all_attributes[j].add(values[i][j])

    medians = []
    for j in range(len(values[0])):
        medians.append(0)
        if type(values[0][j]) == str:
            continue

        sorted_list = []
        for i in range(len(values)):
            sorted_list.append(values[i][j])
        sorted_list.sort()

        medians[j] = sorted_list[int(len(sorted_list)/2)]

    return tree.DecisionTree(
        __recurs_id3(values, labels, gain_function, max_depth, range(len(values[0])), all_attributes, attr_subset_size,
                     medians))


# recursive helper function for id3 function
def __recurs_id3(values, labels, gain_function, max_depth, attributes, attribute_values, attr_subset_size, medians):
    label_frequencies = {}
    # calculate label frequencies
    for label in labels:
        if not (label in label_frequencies):
            label_frequencies[label] = 0
        label_frequencies[label] += 1

    # identify the maximum label
    max_label_freq = 0
    max_label = ""
    for label in label_frequencies.keys():
        if label_frequencies[label] > max_label_freq:
            max_label_freq = label_frequencies[label]
            max_label = label

    # base case: return leaf node
    if max_depth == 0 or len(attributes) == 0 or len(label_frequencies.keys()) == 1:
        return tree.DecisionNode(True, -1, max_label)

    p_values = []
    for label in label_frequencies.keys():
        p_values.append(label_frequencies[label] / len(labels))

    if attr_subset_size > 0:
        attr_subset = random.sample(attributes, min(attr_subset_size, len(attributes)))
    # pick the attribute with maximum gain
    max_gain = -1
    best_attribute = -1
    for attr in attributes:
        # only consider the given attributes
        if attr_subset_size > 0:
            if attr not in attr_subset:
                continue

        # attr_label_frequencies[value] is the frequencies of the labels
        # among the examples with attribute = value
        attr_label_frequencies = {}

        # get the frequencies
        for i in range(len(labels)):
            value = values[i][attr]

            if type(value) != str:
                if value < medians[attr]:
                    value = "less"
                else:
                    value = "more"

            if not (value in attr_label_frequencies):
                attr_label_frequencies[value] = {}

            if not (labels[i] in attr_label_frequencies[value]):
                attr_label_frequencies[value][labels[i]] = 0

            attr_label_frequencies[value][labels[i]] += 1

        # calculate p_v_values and size_values from frequencies
        p_v_values = []
        size_values = []
        for value in attr_label_frequencies.keys():
            p_v_values.append([])

            # number of examples with their attribute equal to value
            num_value = 0
            for label in attr_label_frequencies[value].keys():
                num_value += attr_label_frequencies[value][label]

            size_values.append(num_value / len(labels))

            for label in attr_label_frequencies[value].keys():
                p_v_values[len(p_v_values) - 1].append(attr_label_frequencies[value][label] / num_value)

        # calculate the gain
        gain = 0
        if gain_function == "info gain":
            gain = information_gain(p_values, p_v_values, size_values)

        if gain_function == "maj error":
            gain = majority_error_gain(p_values, p_v_values, size_values)

        if gain_function == "gini":
            gain = gini_index_gain(p_values, p_v_values, size_values)

        if gain > max_gain:
            max_gain = gain
            best_attribute = attr

    # is true if the best attribute is a numeric attribute
    best_attr_numeric = type(values[0][best_attribute]) != str

    if not best_attr_numeric:
        root = tree.DecisionNode(False, best_attribute, "")
    else:
        root = tree.DecisionNode(False, best_attribute, "", is_numeric=True, median=medians[best_attribute])

    # dictionaries that take an attribute value in attr
    # and map to the set S_attr of elements with attribute attr
    values_v = {}
    labels_v = {}
    for attr_value in attribute_values[best_attribute]:
        values_v[attr_value] = []
        labels_v[attr_value] = []

    for i in range(len(labels)):
        value = values[i][best_attribute]
        if best_attr_numeric:
            if value < medians[best_attribute]:
                value = "less"
            else:
                value = "more"

        values_v[value].append(values[i].copy())
        labels_v[value].append(labels[i])

    for attr_value in attribute_values[best_attribute]:
        # if S_V is empty, add a leaf node with most common label
        if len(values_v[attr_value]) == 0:
            root.next_nodes[attr_value] = tree.DecisionNode(True, -1, max_label)
        else:
            root.next_nodes[attr_value] = __recurs_id3(values_v[attr_value], labels_v[attr_value],
                                                       gain_function, max_depth - 1,
                                                       [x for x in attributes if x != best_attribute],
                                                       attribute_values, attr_subset_size, medians)

    return root


# p_values represents the proportion of each label in S
# p_v_values represent the proportion of each label in S_v
# size_values represents the ratios |S_v|/|S| to normalize the gain
def information_gain(p_values, p_v_values, size_values):
    # entropy function
    def h(p_values_):
        result_ = 0.0
        for p in p_values_:
            result_ -= p * np.log2(p)
        return result_

    result = h(p_values)
    for v in range(len(p_v_values)):
        result -= size_values[v] * h(p_v_values[v])

    return result


# p_values represents the proportion of each label in S
# p_v_values represent the proportion of each label in S_v
# size_values represents the ratios |S_v|/|S| to normalize the gain
def majority_error_gain(p_values, p_v_values, size_values):
    # maj error function
    def h(p_values_):
        return 1.0 - max(p_values_)

    result = h(p_values)
    for v in range(len(p_v_values)):
        result -= size_values[v] * h(p_v_values[v])

    return result


# p_values represents the proportion of each label in S
# p_v_values represent the proportion of each label in S_v
# size_values represents the ratios |S_v|/|S| to normalize the gain
def gini_index_gain(p_values, p_v_values, size_values):
    # entropy function
    def h(p_values_):
        result_ = 1.0
        for p in p_values_:
            result_ -= p * p
        return result_

    result = h(p_values)
    for v in range(len(p_v_values)):
        result -= size_values[v] * h(p_v_values[v])

    return result
