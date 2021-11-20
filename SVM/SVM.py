import numpy as np
import random
from scipy.optimize import minimize


# basic learning schedule for default schedule for svm with sgd
def linear_schedule(initial, t):
    return initial / (1 + t)


# runs SVM on the given parameters. Returns a linear classifier which
# can be tested on examples using the .evaluate function.
#
# Values should be an array of float vectors in the same order as the labels,
# which should be an array with values in the set {-1, 1} <- IMPORTANT
#
# num_epochs is the number of epochs SVM is run on. c is the parameter controlling
# the trade off between accuracy and margin size. The loss function we are optimizing is
# L(w) = ||w|| + c * sum max(0, 1 - y_i w * x_i) for all (x_i, y_i) \in (values, labels)
# so the value of c changes our objective function while running SVM. must be positive.
#
# the variable initial_learn is how quickly the algorithm will learn (update w) on each
# iteration at the first iteration. The learning_schedule parameter should be a *function*
# that takes in the initial_learn rate and the number of iterations t and returns
# the next learning rate. should be a function such that sum^infinity learning_schedule(t) diverges
# but sum^infinity learning_schedule(t)^2 converges.
#
# prints information as the algorithm runs if print_info=True.
def svm_sgd(values, labels, num_epochs, c, initial_learn, learning_schedule=linear_schedule, print_info=False):
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
            sub_grad = w.copy()
            sub_grad[len(w)-1] = 0
            if label * np.dot(w, value) < 1:
                sub_grad -= c * len(values) * label * np.array(value)

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


# implementation of svm using the dual form solution. is compatible with kernel trick and therefore
# nonlinear implementations of svm. Returns a linear classifier which
# can be tested on examples using the .evaluate function.
#
# Values should be an array of float vectors in the same order as the labels,
# which should be an array with values in the set {-1, 1} <- IMPORTANT
#
# c is the parameter controlling the trade off between accuracy and margin size. The loss
# function we are optimizing is L(w) = ||w|| + c * sum max(0, 1 - y_i w * x_i) for all
# (x_i, y_i) \in (values, labels) so the value of c changes our objective function
# while running SVM. must be positive.
#
# the kernel should be a function that takes in two float vectors and returns
# a float. The default value is np.dot, which corresponds to when there is no
# non-linearization, so it is the default svm algorithm.
#
# prints information as the algorithm runs if print_info=True.
def svm(values, labels, c, kernel=np.dot, print_info=False):
    values = values.copy()
    labels = labels.copy()

    # to solve the dual form, this is the function we must minimize
    def objective_function(alphas):
        if kernel == np.dot:
            modified_value_matrix = np.matmul(np.matmul(np.diag(labels), np.diag(alphas)), values)
            return 1 / 2 * np.matmul(modified_value_matrix, np.transpose(modified_value_matrix)).sum() - np.sum(alphas)

        kernel_applied_matrix = np.array([kernel(xi, xj) for xi in values for xj in values])
        kernel_applied_matrix = kernel_applied_matrix.reshape((len(values), len(values)))
        diag_matrix = np.matmul(np.diag(labels), np.diag(alphas))
        return 1 / 2 * np.matmul(diag_matrix, np.matmul(kernel_applied_matrix, diag_matrix)).sum() - np.sum(alphas)

    # constraint on the alphas: we must have sum \alpha_i y_i = 0
    def constraint(alphas):
        return np.dot(alphas, labels)

    # bounds for the minimization on alpha
    bounds = [(0, c)] * len(labels)
    constraints = {'type': 'eq', 'fun': constraint}

    x0 = np.array([1]*len(labels))
    solution = minimize(fun=objective_function, x0=x0, method="SLSQP", constraints=constraints, bounds=bounds)
    if not solution.success:
        print("Error with scipy.optimize.minimize:")
        raise Exception(solution.message)

    opt_alphas = solution.x
    num_b_values = 0
    opt_b = 0
    w = np.matmul(np.matmul(np.diag(opt_alphas), np.diag(labels)), values)
    for i in range(len(opt_alphas)):
        if 0 < opt_alphas[i] < c:
            opt_b = labels[i] - np.dot(w, values[i]).sum()
            num_b_values += 1

        opt_b /= num_b_values

    return LinearClassifier(None, kernel=kernel, kernel_vectors=w, bias_term=opt_b)


# A simple class that is returned by the SVM algorithm.
# provides the structure for a standard linear classifier
# as well as a nonlinear classifier using the kernel trick
# to enable the kernel trick, we require two ``types'' of linear classifier,
# one is normal with just a weight vector (with bias included)
# and the other includes a kernel function as well as an array of vectors
class LinearClassifier:

    # initializes a new linear classifier. The default version simply takes in a
    # weight vector "vector" with bias term included. If the kernel trick is
    # to be used, the vector parameter is not needed and may be set to None
    # and the kernel parameter must be a function of two vectors and the
    # kernel_vectors must be a list of vectors x to apply on each example
    # bias_term is incorporated for the non-linear classifier. The bias should be
    # included in the vector for the standard linear classifier.
    def __init__(self, vector, kernel=None, kernel_vectors=None, bias_term=None):
        self.vector = vector
        self.kernel = kernel

        if kernel is not None:
            self.bias_term = bias_term
            self.kernel_vectors = kernel_vectors

    # Evaluates this set of linear classifiers on the given array value
    # we assume the bias term is NOT included
    # returns a value in the set {-1, 1}
    def evaluate(self, value):
        # to include bias term
        _temp_value = value.copy()
        result = 0

        if self.kernel is None:
            _temp_value.append(1)
            result = np.dot(self.vector, _temp_value)
        else:
            for vector in self.kernel_vectors:
                result += self.kernel(vector, _temp_value) + self.bias_term

        if result >= 0:
            return 1
        else:
            return -1
