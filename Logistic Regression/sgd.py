import numpy as np
import random


# basic learning schedule for default schedule for svm with sgd
def linear_schedule(initial, t):
    return initial / (1 + t)


SIGMOID_BOUND = 500


# sigmoid function
def sigmoid(t):
    # to avoid overflow
    if t < -SIGMOID_BOUND:
        return 0
    if t > SIGMOID_BOUND:
        return 1

    return 1 / (1 + np.exp(-t))


# runs linear regression sgd on the given parameters. Returns a linear classifier which
# can be tested on examples using the .evaluate function.
#
# Values should be an array of float vectors in the same order as the labels,
# which should be an array with values in the set {-1, 1} <- IMPORTANT
#
# num_epochs is the number of epochs sgd is run on. variance is the parameter controlling
# the objective function if map is set to true. objective function we are minimizing is
# L(w) = ||w||^2/(2variance) - sum log(sigmoid(y_i w * x_i)) for all (x_i, y_i) \in (values, labels)
# if map = True, and otherwise minimizes L(w) = - sum log(sigmoid(y_i w * x_i)) if map = False
#
# the variable initial_learn is how quickly the algorithm will learn (update w) on each
# iteration at the first iteration. The learning_schedule parameter should be a *function*
# that takes in the initial_learn rate and the number of iterations t and returns
# the next learning rate. should be a function such that sum^infinity learning_schedule(t) diverges
# but sum^infinity learning_schedule(t)^2 converges.
#
# prints information as the algorithm runs if print_info=True.
def lin_regression_sgd(values, labels, num_epochs, variance, initial_learn, learning_schedule=linear_schedule,
                       print_info=False, map=True):
    w = np.array([0.] * (len(values[0]) + 1))
    shuffle_values = []

    t = 0  # total number of iterations thus far

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

        for i in range(len(values)):
            value = values[i].copy()
            label = labels[i]

            # add this term so the bias term is included
            value.append(1)

            # stochastic sub gradient of loss function
            sub_grad = np.array([0.] * (len(values[0]) + 1))
            if map:
                sub_grad = w.copy() / variance
            sub_grad[len(w) - 1] = 0
            if label * np.dot(w, value) < 1:
                sub_grad -= sigmoid(label * np.dot(value, w)) * len(values) * label * np.array(value)

            # update w and t
            w = w - learning_schedule(initial_learn, t) * sub_grad
            if print_info:
                print("values, label, sub-gradient, updated w")
                print(value, end=", ")
                print(label, end=", ")
                print(sub_grad, end=", ")
                print(w)
            t = t + 1

    return LinearClassifier([w])


# A simple class that is returned by the sgd algorithm above.
# provides the structure for a standard linear classifier
class LinearClassifier:

    # initializes a new linear classifier.
    def __init__(self, vector):
        self.vector = vector

    # Evaluates this set of linear classifiers on the given array value
    # we assume the bias term is NOT included
    # returns a value in the set {-1, 1} unless return_avg = True
    def evaluate(self, value, return_avg=False):
        # to include bias term
        _temp_value = value.copy()
        _temp_value.append(1)
        result = np.dot(self.vector, _temp_value)

        if return_avg:
            return result

        if result >= 0:
            return 1
        return -1
