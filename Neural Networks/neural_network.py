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
            if i == layers - 1:
                for j in range(len(self.weights[i])):
                    self.weights[i][j].append(0)
            else:
                for j in range(len(self.weights[i])):
                    for k in range(width):
                        self.weights[i][j].append(0)

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
                self.nodes[i].append(1)
                self.s_values[i].append(1)
            else:
                for j in range(width+1):
                    self.nodes[i].append(1)
                    self.s_values[i].append(1)

    # evaluates the nn at the given value.
    # ALSO temporarily stores the values of each node in the self.nodes matrix
    def evaluate(self, value):
        # resets node values
        for i in range(len(self.nodes)):
            for j in range(len(self.nodes[i])):
                self.s_values[i][j] = 0

        for i in range(len(self.nodes)):
            for node in range(len(self.nodes[i])):
                # no need to compute for bias terms
                if node == len(self.nodes[i]) - 1 and i != len(self.nodes)-1:
                    continue

                if i == 0:
                    for j in range(len(value)):
                        self.s_values[i][node] += self.weights[i][j][node]*value[j]
                    self.s_values[i][node] += self.weights[i][len(value)][node]
                else:
                    for j in range(len(self.nodes[i-1])):
                        self.s_values[i][node] += self.weights[i][j][node]*self.nodes[i-1][j]

                if i != len(self.nodes)-1:
                    self.nodes[i][node] = self.activate(self.s_values[i][node])
                else:
                    self.nodes[i][node] = self.s_values[i][node]

        return self.nodes[len(self.nodes)-1][0] > 0

    # adds a value to each weight
    def increment_weights(self, increment, scalar=1):
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    self.weights[i][j][k] += scalar * increment[i][j][k]
