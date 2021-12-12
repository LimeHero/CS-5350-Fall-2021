ENSEMBLE LEARNING README

The primary function of this code is the ada_boost algorithm
with decision stumps, which aims to learn weighted decision stumps 
from a set of labeled data.

To use the adaboost implementation, one must call

    adaboost.adaboost(values, labels, gain_function, iterations)


The parameters are:
    
    values: an array of attribute arrays. The following is an
    example, where the attributes take values "hot", "cold", "medium"
                    [["cold", "hot", "hot"],
                     ["cold", "hot", "medium"],
                     ["hot", "cold", "medium"]]
    Attributes that are strings are case sensitive. Notice that you can
    also input a numeric value, and the tree will learn based on the median
    of the numeric attribute.

    labels: array of the labels of the data, which the tree attempts to
    learn. Note that the labels must be in the same order as the training values.
    Labels must be strings and are case sensitive. 

    gain_function: must be one of "gini", "maj error", or "info gain".

    iterations: number of decision stumps (iterations) to perform with adaboost

    (optional) print_info: if set to true, will print updates as the training
    progresses

    (optional) test_values: a set of values to test the bagged tree on to measure
    accuracy. only necessary if print_info is true

    (optional) test_labels: labels for test values, only needed if print_info is true

Once the stump forest has been formed by the above syntax, one can use it by calling:

    my_stump_forest.evaluate(values)

with the attribute array values. This will return some label.
If the labels were all numeric, then one can return the average label instead of the
most common among the stumps by setting the optional parameter "avg" to True