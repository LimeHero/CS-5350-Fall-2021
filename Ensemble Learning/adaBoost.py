import numpy as np


# An implementation of a DecisionStump that
# supports weighted training examples.
#
# The constructor of the DecisionStump
# is the id3 algorithm, but where the max depth
# is restricted to 1.
#
# Additionally, it is modified specifically to run the AdaBoost
# algorithm, so it includes a weight matrix D to modify
# the weights of the different training examples.
class DecisionStump:

    # initializes the decision stump such that it chooses the optimal
    # feature to split the data and then splits on that feature.
    #
    # supports weighted training examples, but the numeric attributes
    # should have already been parsed to a numeric - should not be a string.
    def __init__(self, values, labels, gain_function, weights):
        self.map = {}
        if gain_function != "gini" and gain_function != "maj error" and gain_function != "info gain":
            raise Exception("Gain function must be equal to \"gini\", \"maj error\", or \"info gain\"")

        label_frequencies = {}
        # calculate label frequencies
        for label in labels:
            if not (label in label_frequencies):
                label_frequencies[label] = 0
            label_frequencies[label] += 1

        p_values = []
        for label in label_frequencies.keys():
            p_values.append(label_frequencies[label] / len(labels))

        # pick the attribute with maximum gain
        max_gain = -1
        best_attribute = -1
        numeric = [] * len(values[0])
        medians = [] * len(values[0])
        for attr in range(len(values[0])):
            # numeric keeps track of whether this feature is categorical (string) or numerical
            # we assume that strings and numerics are the only attributes passed in
            numeric[attr] = not type(values[0]) == str

            if numeric[attr]:
                sorted_attributes = []
                for i in range(len(labels)):
                    sorted_attributes.append(values[i][attr])
                sorted_attributes.sort()
                medians[attr] = sorted_attributes[int(len(sorted_attributes) / 2)]

            # attr_label_frequencies[value] is the frequencies of the labels
            # among the examples with attribute = value
            attr_label_frequencies = {}

            # get the (weighted) frequencies
            for i in range(len(labels)):
                value = values[i][attr]
                if numeric[attr]:
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

                # number of (weighted) examples with their attribute equal to value
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

        self.feature = best_attribute  # self.feature is this decision stumps feature
        self.is_numeric = numeric[best_attribute]
        self.median = medians[best_attribute]

        # count the frequency of the labels on each value best_attribute can take
        best_attr_label_frequencies = {}
        for i in range(len(labels)):
            value = values[i][best_attribute]
            if self.is_numeric:
                if value < self.median:
                    value = "less"
                else:
                    value = "more"

            if not (value in best_attr_label_frequencies):
                best_attr_label_frequencies[value] = {}

            if not (labels[i] in best_attr_label_frequencies[value]):
                best_attr_label_frequencies[value][labels[i]] = 0

            best_attr_label_frequencies[value][labels[i]] += 1

        # complete the stump so each leaf is the most frequent label in that subtree
        for value in best_attr_label_frequencies.keys():
            most_freq_label = ""
            freq = -1
            for label in best_attr_label_frequencies[value].keys():
                if best_attr_label_frequencies[values][label] > freq:
                    most_freq_label = label

            self.map[value] = most_freq_label

    # Returns this decision stump evaluated on the given list of features
    def evaluate(self, values):
        if not self.is_numeric:
            return self.map[values[self.feature]]

        if values[self.feature] < self.median:
            return self.map["less"]

        return self.map["more"]


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
