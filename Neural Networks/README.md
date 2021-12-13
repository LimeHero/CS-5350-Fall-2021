NEURAL NETWORK README

This code implements the back propagation algorithm on a basic fully
connected neural network. The back propagation algorithm implements
stochastic gradient descent with adjustable learning rate and schedule.
To use this implementation of neural network learning by back propagation,
use the following syntax:

    nn_sgd.nn_sgd(values, labels, convergence, [layers, width], initial_learn)

The parameters are:

    values: an array of float vectors to be used as training values. 
    the vectors should all be of the same dimension.

    labels: corresponding labels to the training values in the same
    order. labels must be floats, and preferably in the range [-1,1]

    convergence: sgd will run until the difference in loss between
    iterations reaches "convergence" in absolute value. Thus,
    convergence must be positive and the algorithm will run longer
    the smaller convergence is.

    [layers, width]: the layers and width of the neural network 
    (not including bias terms). by default, the neural network will
    have the same width in each layer.

    initial_learn: the learning rate is the parameter sometimes 
    called 'r', which denotes how much the weight vector is updated 
    on each mistake. initial_learn is the initial value of this value

    (optional) learning_schedule: the schedule of how the learning rate
    changes on each step of the algorithm. default is linear, so the
    learning rate at step t is r_0 / (t + 1), r_0 is initial learning rate.

    (optional) activation_function: the activation function of the
    neural network. default value is the sigmoid function

    (optional) activation_derivative: the derivative of the activation
    function. must be changed with the activation_function for proper
    behavior. default is the derivative of the sigmoid function.

    (optional) print_info: will print the loss at each epoch if set
    to True. default value is False.


