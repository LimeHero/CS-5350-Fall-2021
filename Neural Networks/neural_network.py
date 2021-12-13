import numpy as np


# A class for a fully connected neural network
# with an arbitrary number of layers, width (number of nodes in each layer)
# and input_width (width of input vector / base layer). will return a float
#
# layers > 0, width > 0, input_width > 0 should be satisfied.
# layers = 1 corresponds to a linear classifier.
# notice that input_width and width do NOT include the bias term
#
#
# self.weights is a 3-tensor such that self.weights[i][j][k] is the edge
# connecting layer i-1 to layer i from node j to node k (w^{i+1}_{jk})
class NeuralNetwork:

    # constructs a new neural network with the given architecture,
    # activation function, and all parameters (weights) equal to 0
    def __init__(self, layers, width, input_width, activation_func):
        self.activate = activation_func

        self.weights = []
        # initialize architecture
        for i in range(layers):
            self.weights.append([])

            # base layer has different width (input_width)
            if i == 0:
                # add the bias term (the + 1)
                for j in range(input_width + 1):
                    self.weights[i].append([])
            else:
                for j in range(width + 1):
                    self.weights[i].append([])

            # final layer has different width (1)
            for j in range(len(self.weights[i])):
                if i == layers - 1:
                    self.weights[i][j].append(0.)
                else:
                    for k in range(width):
                        self.weights[i][j].append(0.)
            self.weights[i] = np.array(self.weights[i])
        self.weights = np.array(self.weights, dtype=object)

        # when evaluate is called, values of each node are temporarily
        # stored in self.nodes to (greatly) reduce computation when learning
        self.nodes = []

        # when evaluate is called, values of each node BEFORE activation function is called
        # are temporarily stored in self.s_values to reduce computation time in
        # computing back propagation
        self.s_values = []
        for i in range(layers):
            self.nodes.append([])
            self.s_values.append([])

            # final result node (output)
            if i == layers - 1:
                self.nodes[i].append(1.)
                self.s_values[i].append(1.)
            else:
                for j in range(width):
                    self.nodes[i].append(1.)
                    self.s_values[i].append(1.)
                self.nodes[i].append(1.)  # bias term

            self.nodes[i] = np.array(self.nodes[i])
            self.s_values[i] = np.array(self.s_values[i])

    # evaluates the nn at the given value.
    # ALSO temporarily stores the values of each node in the self.nodes matrix
    def evaluate(self, value):
        for layer in range(len(self.nodes)):
            for node in range(len(self.s_values[layer])):
                # resets node values
                self.s_values[layer][node] = 0.0

        value_with_bias = value.copy()
        value_with_bias.append(1)
        value_with_bias = np.array(value_with_bias)

        for layer in range(len(self.nodes)):
            if layer == 0:
                self.s_values[layer] = np.matmul(value_with_bias, self.weights[layer])
            else:
                self.s_values[layer] = np.matmul(self.nodes[layer-1], self.weights[layer])

            for node in range(len(self.s_values[layer])):
                if layer != len(self.nodes)-1:
                    self.nodes[layer][node] = self.activate(self.s_values[layer][node])
                else:
                    self.nodes[layer][node] = self.s_values[layer][node]

        return self.nodes[len(self.nodes)-1][0]

    # adds a value to each weight
    def increment_weights(self, increment, scalar=1):
        self.weights += scalar * increment
