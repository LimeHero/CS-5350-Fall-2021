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
    Attributes that are strings are case sensitive. Notice that you can
    also input a numeric value, and the tree will learn based on the median
    of the numeric attribute.

    train_labels: array of the labels of the data, which the tree attempts to
    learn. Note that the labels must be in the same order as the training values.
    Labels must be strings and are case sensitive.

    gain_function: must be one of "gini", "maj error", or "info gain".

    max_depth: the maximum depth of the tree. Must be an integer >= 0. 
    If max_depth is equal to 1, the algorithm will return a decision stump.

To use a DecisionTree object (returned from id3), use the following syntax:

    tree.evaluate(values)

where values is an array of data attributes, such as ["cold", "hot", "hot"].

To use a bagged tree, first initialize a blank bagged tree object, and then
run initialize_id3 with the following syntax:
    
    bagged_tree.BaggedTree()
    bagged_tree.initialize_id3(train_values, train_labels, gens, gain_function, m)

The parameters are:
    
    train_values: an array of attribute arrays. The following is an
    example, where the attributes take values "hot", "cold", "medium"
                    [["cold", "hot", "hot"],
                     ["cold", "hot", "medium"],
                     ["hot", "cold", "medium"]]
    Attributes that are strings are case sensitive. Notice that you can
    also input a numeric value, and the tree will learn based on the median
    of the numeric attribute.

    train_labels: array of the labels of the data, which the tree attempts to
    learn. Note that the labels must be in the same order as the training values.
    Labels must be strings and are case sensitive.

    gens: number of trees to add to the final bagged tree result    

    gain_function: must be one of "gini", "maj error", or "info gain".

    m: number of randomly selected training examples used to train each tree

    (optional) attr_subset_size: if set, changes the bagged tree to a random forest.
    this option will change the trees learned so they only learn from a random attr_subset_size 
    subset of the attributes at each iteration

    (optional) print_info: if set to true, will print updates as the training
    progresses

    (optional) test_values: a set of values to test the bagged tree on to measure
    accuracy. only necessary if print_info is true

    (optional) test_labels: labels for test values, only needed if print_info is true

    (optional) all_attributes: so the trees are well formed, if the train_examples
    do not contain all possible attribute values

Once the bagged tree has been formed by the above syntax, one can use it by calling:

    my_bagged_tree.evaluate(values)

with the attribute array values. This will return some label.
If the labels were all numeric, then one can return the average label instead of the
most common among the trees by setting the optional parameter "avg" to True