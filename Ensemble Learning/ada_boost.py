import numpy as np


# Runs ada_boost with the given values and labels
# and returns a stump forest with "iterations" number of stumps.
#
# gain_function must be one of "gini", "maj error", or "info gain"
#
# if print_info is True, train_values and train_labels must be provided,
# in which case ada_boost will print the train and test accuracies on each generation
# along with the individual stump weighted accuracy and test accuracy
def ada_boost(values, labels, gain_function, iterations, print_info=False, test_values=None, test_labels=None):
    if gain_function != "gini" and gain_function != "maj error" and gain_function != "info gain":
        raise Exception("Gain function must be equal to \"gini\", \"maj error\", or \"info gain\"")

    if print_info:
        print("iteration,stump weighted error,stump test error,training error,test error,")
    stump_forest = StumpForest()
    weights = [1] * len(values)
    for i in range(iterations):
        if print_info:
            print(i, end=",")
        next_stump = DecisionStump()
        vote_const = next_stump.ada_boost_initialization(values, labels, gain_function, weights, print_info=print_info)
        stump_forest.add_stump(next_stump, vote_const)

        if print_info:
            # stump test error
            num_right = 0
            for j in range(len(test_values)):
                if next_stump.evaluate(test_values[j]) != test_labels[j]:
                    num_right += 1
            print(num_right / len(test_values), end=",")

            # train error
            num_right = 0
            for j in range(len(values)):
                if stump_forest.evaluate(values[j]) != labels[j]:
                    num_right += 1
            print(num_right / len(values), end=",")

            # test error
            num_right = 0
            for j in range(len(test_values)):
                if stump_forest.evaluate(test_values[j]) != test_labels[j]:
                    num_right += 1
            print(num_right / len(test_values))

    return stump_forest


# a collection of stumps with weighted votes,
# used as the output of adaboost
class StumpForest:
    def __init__(self):
        self.weights = []
        self.stumps = []

    # add a stump to the forest
    def add_stump(self, stump, weight):
        self.weights.append(weight)
        self.stumps.append(stump)

    # evaluates the weighted vote of the stumps
    # must be a categorical label in this implementation
    # print_avg is used *ONLY* if the labels are numerical
    # and we wish to return the average label value instead of most frequent
    def evaluate(self, values, print_avg=False):
        if print_avg:
            label_sum = 0
            for i in range(len(self.stumps)):
                label_sum += self.stumps[i].evaluate(values)
            return label_sum / len(self.stumps)

        label_count = {}
        for i in range(len(self.stumps)):
            label = self.stumps[i].evaluate(values)

            if label not in label_count:
                label_count[label] = 0

            label_count[label] += self.weights[i]

        max_label_count = 0
        best_label = ""
        for label in label_count.keys():
            if label_count[label] > max_label_count:
                best_label = label
                max_label_count = label_count[label]

        return best_label


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

    # initializes the decision stump with empty variables
    # note that the decision stump is *not* well formed after
    # this initialization.
    #
    # This class is built to use in ada_boost, and so the initialization
    # of the stump should be ran in ada_boost_initialization
    def __init__(self):
        self.map = {}  # defines the "leaf nodes" of this stump, so map[attribute] = label
        self.feature = None  # the feature (integer) this stump splits on
        self.is_numeric = False  # whether or not the feature is numeric
        self.median = None  # if the feature is numeric, this is the median training value

    # in the given values and labels, selects the best
    # feature to split the data and then splits on that feature.
    #
    # supports weighted training examples, but the numeric attributes
    # should have already been parsed to a numeric - should not be a string.
    #
    # the weights array changes how much each training example should be weighted
    # so the ith training example is counted weights[i] times
    #
    # note that the weights array should be normalized so that the sum
    # is equal to len(values), instead of 1. This is to save some unnecessary
    # possible rounding error
    #
    # returns the "vote constant" for this stump, which is defined to be the value
    # .5 * ln((1-err)/err) where err is the error computed for this stump.
    #
    # additionally, updates the weight array as necessary
    def ada_boost_initialization(self, values, labels, gain_function, weights, print_info=False):
        if gain_function != "gini" and gain_function != "maj error" and gain_function != "info gain":
            raise Exception("Gain function must be equal to \"gini\", \"maj error\", or \"info gain\"")

        label_frequencies = {}
        # calculate label frequencies
        for i in range(len(labels)):
            label = labels[i]
            if not (label in label_frequencies):
                label_frequencies[label] = 0
            label_frequencies[label] += weights[i]

        p_values = []
        for label in label_frequencies.keys():
            p_values.append(label_frequencies[label] / len(labels))

        # pick the attribute with maximum gain
        max_gain = -1
        best_attribute = -1
        numeric = [False] * len(values[0])
        medians = [0] * len(values[0])
        for attr in range(len(values[0])):
            # numeric keeps track of whether this feature is categorical (string) or numerical
            # we assume that strings and numerics are the only attributes passed in
            numeric[attr] = not type(values[0][attr]) == str

            # get median if numeric attribute
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

                attr_label_frequencies[value][labels[i]] += weights[i]  # weighted sum

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
                    p_v_values[-1].append(attr_label_frequencies[value][label] / num_value)

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

            best_attr_label_frequencies[value][labels[i]] += weights[i]

        # complete the stump so each leaf is the most frequent label in that subtree
        for value in best_attr_label_frequencies.keys():
            most_freq_label = ""
            freq = -1
            for label in best_attr_label_frequencies[value].keys():
                if best_attr_label_frequencies[value][label] > freq:
                    most_freq_label = label
                    freq = best_attr_label_frequencies[value][label]

            self.map[value] = most_freq_label

        # compute the error and vote constant
        error = 0
        for i in range(len(labels)):
            if labels[i] != self.evaluate(values[i]):
                error += weights[i]

        error /= len(labels)
        if print_info:
            print(error, end=",")
        vote_constant = np.log((1 - error) / error) / 2

        # update the weights array
        for i in range(len(weights)):
            if labels[i] != self.evaluate(values[i]):
                weights[i] *= np.e ** vote_constant
            else:
                weights[i] *= np.e ** (-vote_constant)

        norm = len(labels)/sum(weights)
        for i in range(len(weights)):
            weights[i] *= norm

        return vote_constant

    # Returns this decision stump evaluated on the given list of features
    def evaluate(self, values):
        if not self.is_numeric:
            # if its not well formed, just pick the first one
            if values[self.feature] not in self.map.keys():
                return self.map[list(self.map.keys())[0]]
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
