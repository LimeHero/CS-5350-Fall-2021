PERCEPTRON README

This code implements the perceptron algorithm. 
To use this implementation, call:

    perceptron.perceptron(values, labels, num_epochs, learning_rate)

The parameters are:

    values: an array of float vectors to be used as training values. 
    the vectors should all be of the same dimension.

    labels: corresponding labels to the training values in the same
    order. labels MUST be an array of the values {-1, 1}. 

    num_epochs: number of epochs to run perceptron on.

    learning_rate: the parameter sometimes called 'r', which denotes
    how much the weight vector is updated on each mistake.

    (optional) margin: if margin > 0, then the algorithm will count
    test examples close to the margin as mistakes and update accordingly.
    default value is 0 (no margin).

    (optional) perc_type: the type of perceptron to be formed. must be one
    of "standard", "vote", or "average". "standard" is the basic perceptron
    algorithm, "vote" will create a voted linear classifier, and "average"
    is functionally equivalent to "vote" but faster. default value is "average".

    (optional) print_info: if true, will print information as the algorithm
    runs. 

A LinearClassifier object will be returned. It can be evaluated on test examples
using the function 

    lin_classifier.evaluate(value)

This function will return 1 or -1. 