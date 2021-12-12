LINEAR REGRESSION README

The primary function of this code is to find the least mean squares
vector for a given data set using gradient descent and stochastic
gradient descent. 

To use the implementation for this problem, call:
    
    lms.lms(train_values, train_labels, r, descent_type)

The parameters are:

    train_values: an array of floats that are used to train on. should all be 
    vectors in the same dimension.

    train_labels: labels associated with the train_values, in the same order.

    r: learning rate.

    descent_type: must be one of "sgd" or "grad", for stochastic and regular
    gradient descent. Default value is "sgd".

    (optional) print_info: will print updates if set to true

A vector will be returned, which is what the algorithm found to be the optimal 
weight vector.