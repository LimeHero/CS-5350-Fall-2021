DECISION TREE README

The primary function of this code is the implementation of the id3
algorithm, which aims to learn an optimal decision tree from
a set of labeled data. 

To use the id3 implementation, one must call

    id3.id3(train_values, train_labels, gain_function, max_depth)

This will return a *DecisionTree* object according to the id3 algorithm.
The parameters are:

    train_values: an array of attribute arrays. The following is an
    example, where the attributes take values "hot", "cold", "medium"
                    [["cold", "hot", "hot"],
                     ["cold", "hot", "medium"],
                     ["hot", "cold", "medium"]]
    Attributes must be strings and are case sensitive.

    train_labels: array of the labels of the data, which the tree attempts to
    learn. Note that the labels must be in the same order as the training values.
    Labels must be strings and are case sensitive.

    gain_function: must be one of "gini", "maj error", or "info gain".

    max_depth: the maximum depth of the tree. Must be >= 0. If max_depth
    is equal to 1, the algorithm will return a decision stump.

To use a DecisionTree object (returned from id3), use the following syntax:

    tree.evaluate(values)

where values is an array of data attributes, such as ["cold", "hot", "hot"].