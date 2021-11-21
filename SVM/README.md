SVM README

This code implements the SVM algorithm using the dual
solution as well as a stochastic sub-gradient solution.
To use this implementation of the stochastic sub-gradient solution
to SVM, use the following function call:

    SVM.svm_sgd(values, labels, num_epochs, c, initial_learn)

The parameters are:

    values: an array of float vectors to be used as training values. 
    the vectors should all be of the same dimension.

    labels: corresponding labels to the training values in the same
    order. labels MUST be an array of the values {-1, 1}. 

    num_epochs: number of epochs to run perceptron on.

    c: a model parameter that changes how much SVM will value 
    training accuracy (higher value of c) versus more functional
    margin (lower c)

    initial_learn: the learning rate is the parameter sometimes 
    called 'r', which denotes how much the weight vector is updated 
    on each mistake. initial_learn is the initial value

    (optional) learning_schedule: the schedule of how the learning rate
    changes on each step of the algorithm. default is linear, so the
    learning rate at step t is r_0 / (t + 1), r_0 is initial learning rate.

To use the SVM algorithm with the dual implementation (referred),
use the following function call:

    SVM.svm(values, labels, c, kernel=np.dot)

The parameters are:

    values: an array of float vectors to be used as training values. 
    the vectors should all be of the same dimension.

    labels: corresponding labels to the training values in the same
    order. labels MUST be an array of the values {-1, 1}. 

    c: a model parameter that changes how much SVM will value 
    training accuracy (higher value of c) versus more functional
    margin (lower c)

    (optional) kernel: used as the function for the kernel trick.
    should be a function that takes in two vectors and returns 
    a scalar (i.e., dot product). used to provide non-linearization
    to SVM. default is np.dot. 