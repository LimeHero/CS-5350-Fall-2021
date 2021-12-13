import numpy as np
import random
import neural_network


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


# derivative of sigmoid function
def sigmoid_derivative(t):
    # to avoid overflow
    if t < -SIGMOID_BOUND or t > SIGMOID_BOUND:
        return 0

    return sigmoid(t) * (1 - sigmoid(t))


# learns a neural network on the given parameters. Returns a neural network
# can be tested on examples using the .evaluate function.
#
# Values should be an array of float vectors in the same order as the labels,
# which should be an array of the elements [-1,1].
#
# convergence is the value that the difference of the objective function must reach to terminate.
# The loss function we are optimizing is least mean squares, so sum (y-y^i)^2.
#
# the nn_dimensions parameter should be a an array of the form [a,b], where a is the width
# of the neural network and b is the number of layers
#
# the variable initial_learn is how quickly the algorithm will learn (update weights) on each
# iteration at the first iteration. The learning_schedule parameter should be a *function*
# that takes in the initial_learn rate and the number of iterations t and returns
# the next learning rate. should be a function such that sum^infinity learning_schedule(t) diverges
# but sum^infinity learning_schedule(t)^2 converges (i.e., 1/t).
#
# The activation function can be configured, but the default is the sigmoid function.
# If the activation function is modified, one MUST also modify the activation derivative function.
#
# prints information as the algorithm runs if print_info=True.
def nn_sgd(values, labels, convergence, nn_dimensions, initial_learn, learning_schedule=linear_schedule,
           activation_function=sigmoid, activation_derivative=sigmoid_derivative, print_info=False):
    layers = nn_dimensions[0]
    width = nn_dimensions[1]

    nn = neural_network.NeuralNetwork(layers, width, len(values[0]), activation_function)

    # choose nn weights
    for i in range(len(nn.weights)):
        for j in range(len(nn.weights[i])):
            for k in range(len(nn.weights[i][j])):
                nn.weights[i][j][k] = np.random.normal()

    # we make node_gradient and weight_gradient once so we do not need to remake
    # on every pass of back propagation
    node_gradient = []
    for i in range(layers):
        node_gradient.append([])
        if i == len(nn.nodes) - 1:
            node_gradient[i].append(0.)
        else:
            for j in range(width):
                node_gradient[i].append(0.)
        node_gradient[i] = np.array(node_gradient[i])
    node_gradient = np.array(node_gradient, dtype=object)

    weight_gradient = []
    for i in range(len(nn.weights)):
        weight_gradient.append([])
        for j in range(len(nn.weights[i])):
            weight_gradient[i].append([])
            for k in range(len(nn.weights[i][j])):
                weight_gradient[i][j].append(0.)
            weight_gradient[i][j] = np.array(weight_gradient[i][j])
        weight_gradient[i] = np.array(weight_gradient[i])
    weight_gradient = np.array(weight_gradient, dtype=object)

    prev_loss = -1

    t = 0  # total number of updates thus far

    # to shuffle learning data
    shuffle_values = []

    for i in range(len(values)):
        shuffle_values.append(i)

    for epoch in range(100_000):

        loss = 0
        for i in range(len(labels)):
            loss += (nn.evaluate(values[i]) - labels[i]) ** 2 / 2

        if print_info:
            print("Current Epoch: " + str(epoch))
            print("loss: " + str(loss))
            print()

        if np.abs(loss - prev_loss) < convergence and prev_loss >= 0:
            break

        prev_loss = loss

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

        for i in range(len(labels)):
            back_propagation(nn, values[i], labels[i], activation_derivative,
                             nn_dimensions, node_gradient, weight_gradient)

            nn.increment_weights(weight_gradient, scalar=-learning_schedule(initial_learn, t))

            t += 1

    return nn


# updates the neural network by stochastic gradient descent with a test pair (value, label),
# with learning rate of learn_rate. nn_dimensions stores the dimensions of the neural network
# as [layers, width]. activation_derivative is a function \sigma' that should return the derivative
# of the activation function (at some x).
def back_propagation(nn, value, label, activation_derivative, nn_dimensions, node_gradient,
                     weight_gradient):
    layers = nn_dimensions[0]
    width = nn_dimensions[1]

    for i in range(len(node_gradient)):
        node_gradient[i].fill(0.)

    for i in range(len(weight_gradient)):
        weight_gradient[i].fill(0.)

    # print(node_gradient)
    node_gradient[layers - 1][0] = nn.evaluate(value) - label

    # since the final and first layer are different, we need a totally different case
    # for when these are the same layer
    if layers == 1:
        for i in range(len(value)):
            weight_gradient[0][i][0] = node_gradient[0][0] * value[i]
        weight_gradient[0][len(value)][0] = node_gradient[0][0]

    for layer in range(layers - 1, -1, -1):
        if layers == 1:
            break

        # since activation function not applied on output, a little different for last layer:
        partial_derivative = []
        if layer == layers - 1:
            partial_derivative = [node_gradient[layer][0]]
        else:
            for j in range(width):
                partial_derivative.append(node_gradient[layer][j] * activation_derivative(nn.s_values[layer][j]))

        # we don't need to calculate node_gradient on the last step
        if layer == 0:
            value_with_bias = value.copy()
            value_with_bias.append(1)
            value_with_bias = np.array(value_with_bias)

            weight_gradient[layer] = np.kron(value_with_bias, partial_derivative).reshape((len(value_with_bias), width))

            continue

        shape = (width + 1, width)
        if layer == layers - 1:
            shape = (width + 1, 1)

        weight_gradient[layer] = np.kron(nn.nodes[layer - 1], partial_derivative).reshape(shape)

        node_gradient[layer - 1] = np.matmul(nn.weights[layer], partial_derivative)[:-1]
