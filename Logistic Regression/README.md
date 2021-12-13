LOGISTIC REGRESSION README

This code implements a stochastic gradient descent solution to
logistic regression. In particular, this code implements a sgd solution
for the Maximum a Posteriori (ML) and Maximum Likelihood (ML) objective
functions to logistic regression (i.e., functions \R^n \to [-1,1]). To
use this implementation of logistic regression, use the following
syntax:

    lin_regression_sgd(values, labels, num_epochs, variance, initial_learn)

The parameters are:

    values: an array of float vectors to be used as training values. 
    the vectors should all be of the same dimension.

    labels: corresponding labels to the training values in the same
    order. labels MUST be an array of the values {-1, 1}. 

    num_epochs: number of epochs to run sgd on.

    variance: a model parameter which changes the objective function
    if MAP is used. if ML is used, this value has no effect.

    initial_learn: the learning rate is the parameter sometimes 
    called 'r', which denotes how much the weight vector is updated 
    on each mistake. initial_learn is the initial value

    (optional) learning_schedule: the schedule of how the learning rate
    changes on each step of the algorithm. default is linear, so the
    learning rate at step t is r_0 / (t + 1), r_0 is initial learning rate.

    (optional) print_info: will print information on each iteration if
    set to true. default is False.

    (optional) map: if true, will run with MAP objective function. 
    if false, will run with ML objective function. default is true.