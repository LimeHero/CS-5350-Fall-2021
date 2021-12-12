import numpy as np
import random
import neural_network


# basic learning schedule for default schedule for svm with sgd
def linear_schedule(initial, t):
    return initial / (1 + t)


# sigmoid function
def sigmoid(t):
    return 1 / (1 + np.exp(-t))


# derivative of sigmoid function
def sigmoid_derivative(t):
    return sigmoid(t) * (1 - sigmoid(t))


# learns a neural network on the given parameters. Returns a neural network
# can be tested on examples using the .evaluate function.
#
# Values should be an array of float vectors in the same order as the labels,
# which should be an array of the elements [-1,1].
#
# num_epochs is the number of epochs we will learn on. The loss function we are optimizing is
# least mean squares, so sum (y-y^i)^2.
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
def nn_sgd(values, labels, num_epochs, nn_dimensions, initial_learn, learning_schedule=linear_schedule,
           activation_function=sigmoid, activation_derivative=sigmoid_derivative, print_info=False):
    nn = neural_network.NeuralNetwork(nn_dimensions[0], nn_dimensions[1], len(values[0]), activation_function)

    random_weights = nn.weights

    for i in range(len(random_weights)):
        for j in range(len(random_weights[i])):
            for k in range(len(random_weights[i][j])):
                random_weights[i][j][k] = (len(random_weights)-i)*(2*random.random() - 1)

    t = 0  # total number of updates thus far

    # to shuffle learning data
    shuffle_values = []

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

        for i in range(len(labels)):
            back_propagation(nn, values[i], labels[i], learning_schedule(initial_learn, t), activation_derivative,
                             nn_dimensions)

            t = t + 1

        if print_info:
            print("Current Epoch: " + str(epoch))
            incorrect_count = 0
            for i in range(len(labels)):
                if nn.evaluate(values[i]) != labels[i]:
                    incorrect_count += 1

            print("training error: " + str(incorrect_count / len(labels)))
            print()

    return nn


# updates the neural network by stochastic gradient descent with a test pair (value, label),
# with learning rate of learn_rate. nn_dimensions stores the dimensions of the neural network
# as [layers, width]. activation_derivative is a function \sigma' that should return the derivative
# of the activation function (at some x).
def back_propagation(nn, value, label, learn_rate, activation_derivative, nn_dimensions):
    layers = nn_dimensions[0]
    width = nn_dimensions[1]

    node_gradient = []
    for i in range(layers):
        node_gradient.append([])
        if i == len(nn.nodes) - 1:
            node_gradient[i].append(nn.evaluate(value) - label)  # evaluate neural network on value
        else:
            for j in range(width):
                node_gradient[i].append(0)

    weight_gradient = []
    for i in range(len(nn.weights)):
        weight_gradient.append([])
        for j in range(len(nn.weights[i])):
            weight_gradient[i].append([])
            for k in range(len(nn.weights[i][j])):
                weight_gradient[i][j].append(0)

    # since the final and first layer are different, we need a totally different case
    # for when these are the same layer
    if layers == 1:
        for i in range(len(value)):
            weight_gradient[0][i][0] = node_gradient[0][0]*value[i]
        weight_gradient[0][len(value)][0] = node_gradient[0][0]

    for layer in range(layers-1, -1, -1):
        if layers == 1:
            break

        # since activation function not applied on output, a little different for last layer:
        if layer == layers-1:
            for i in range(len(weight_gradient[layer])):
                # only one node in final layer, so only need nodes[layer][0]
                weight_gradient[layer][i][0] = node_gradient[layer][0]*nn.nodes[layer-1][i]

            for i in range(width):
                node_gradient[layer-1][i] = node_gradient[layer][0]*nn.weights[layer][i][0]

            continue

        for j in range(width):
            # the value \partial L / \partial node_j^k * \si'(node_j^k) which is needed frequently
            partial_derivative_j = node_gradient[layer][j]*activation_derivative(nn.s_values[layer][j])
            # at the base layer we work with the input vector so its a little different
            if layer == 0:
                for i in range(len(value)):
                    weight_gradient[layer][i][j] = partial_derivative_j*value[i]
                weight_gradient[layer][len(value)][j] = partial_derivative_j  # bias term

                continue

            # this is the "normal" case
            for i in range(width + 1):
                weight_gradient[layer][i][j] = partial_derivative_j*nn.nodes[layer-1][i]

                if i == width:
                    continue

                node_gradient[layer-1][i] += partial_derivative_j*nn.weights[layer][i][j]

    nn.increment_weights(weight_gradient, scalar=-learn_rate)







