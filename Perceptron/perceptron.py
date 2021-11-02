import numpy as np
import random


# Runs perceptron with the provided values and labels for the given number
# of epochs. Returns a LinearClassifier object, which is a simple
# container class to help return the output from a (voted) linear classifier
#
# The values and labels arrays must be in the same order, so values[i] has label
# labels[i]. Values must be an array of floats (numerics). No need to include
# the modified values which include 1 as their last entry
# Labels must be an array only with values {-1, 1}. NOT {0, 1}.
#
# The learning rate is the parameter r, which is how much perceptron updates on
# each mistake
#
# margin is an optional attribute which causes perceptron to record an error when
# it has a prediction close to the margin of the linear classifier
#
# perc_type refers to the type of perceptron to be performed: must be one of
# "average", "vote", or "standard". Default is "average"
#
# If print_info is true, will return some information as the algorithm progresses
# depending on the perc_type used
def perceptron(values, labels, num_epochs, learning_rate, margin=0, perc_type="average", print_info=False):
    if perc_type != "average" and perc_type != "vote" and perc_type != "standard":
        raise Exception("Gain function must be equal to \"average\", \"vote\", or \"standard\"")

    w = np.array([0.] * (len(values[0])+1))
    shuffle_values = []

    votes = []  # only used if perc_type == "vote"
    prev_w = []  # only used if perc_type == "vote"
    correct_count = 0  # only used if perc_type == "vote"

    average = np.array([0.] * len(w))  # only used if perc_type == "average"

    for i in range(len(values)):
        shuffle_values.append(i)

    for epoch in range(num_epochs):

        # shuffle the training data
        random.shuffle(shuffle_values)
        _temp_values = []
        _temp_labels = []

        for i in range(len(values)):
            _temp_values.append(values[shuffle_values[i]].copy())
            _temp_labels.append(labels[shuffle_values[i]])

        for i in range(len(values)):
            values[i] = _temp_values[i]
            labels[i] = _temp_labels[i]

        # go through the training set
        for i in range(len(values)):
            value = values[i].copy()
            label = labels[i]

            # add this term so the bias term is included
            value.append(1)

            # mistake made, so update
            if label*np.dot(value, w) <= margin:
                if perc_type == "vote":
                    if not (epoch == 0 and i == 0):
                        votes.append(correct_count)
                        prev_w.append(w)
                        correct_count = 0

                w = w + (learning_rate*label)*np.array(value)

            if perc_type == "vote":
                correct_count += 1

            if perc_type == "average":
                average += w

    if perc_type == "vote":
        votes.append(correct_count)
        prev_w.append(w)
        if print_info:
            print("Weight vector and vote list: ")
            print("w_1 & w_2 & w_3 & w_4 & w_5 & vote_count")
            weight_sum = [0.] * len(w)
            for i in range(len(votes)):
                weight_sum += votes[i]*prev_w[i]
                for j in range(len(prev_w[i])):
                    formatted = format(prev_w[i][j], '.5f')
                    print(formatted + " & ", end="")

                print(str(votes[i]) + " \\\\ \n \\hline ")
            print("Weighted sum of all weight vectors: " + str(weight_sum))

        return LinearClassifier(prev_w, votes=votes)

    if perc_type == "average":
        if print_info:
            print("Final (average) weight vector: " + str(average))
        return LinearClassifier([average])

    if print_info:
        print("Final weight vector: " + str(w))
    return LinearClassifier([w])


# A simple class that is returned by the perceptron algorithm.
# provides the structure for a weighted linear classifier (i.e., what
# is returned by a voted perceptron) as well as for a standard linear classifier.
class LinearClassifier:

    # Initializes a linear classifier with the given vectors (list of linear classifiers)
    # and votes if desired. If no votes is specified, will be all 1's.
    def __init__(self, vectors, votes=None):
        self.vectors = vectors

        if votes is None:
            self.weights = [1] * len(vectors)
        else:
            self.weights = votes

    # Evaluates this set of linear classifiers on the given array value
    # we assume the bias term is NOT included
    # returns a value in the set {-1, 1}
    def evaluate(self, value):
        # to include bias term
        _temp_value = value.copy()
        _temp_value.append(1)
        result = 0
        for i in range(len(self.vectors)):
            result += self.weights[i] * np.dot(self.vectors[i], _temp_value)

        if result >= 0:
            return 1
        else:
            return -1
