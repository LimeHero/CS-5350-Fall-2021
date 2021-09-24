# Class that produces the behavior of a decision tree.
# In particular, every DecisionNode is either a leaf node
# or not.
#
# In the case that this node is a leaf node,
# it is effectively storing a string corresponding to a label.
#
# In the case that this node is not a leaf node, it has an integer i
# corresponding to which attribute it is storing, and then
# maps to other nodes based on the value of that attribute. For instance,
# if attribute = 0 and our data is such that the 0th attribute is
# eye color, this node's dictionary may have key values "blue", "green", etc.
class DecisionNode:

    # Initializes a node which is root by default.
    # If this node is not a leaf node, initializes the node
    # with the given attributes.
    #
    # If this node is a leaf node, initializes it with the given label.
    def __init__(self, is_leaf_node, attribute_int, label):
        # next_nodes is a dictionary from attribute values to other nodes in the tree
        # in the case that this node is a leaf node, next_nodes is empty
        self.next_nodes = {}

        # stores which attribute this (non-leaf) node is tracking
        self.attribute = 0

        # if this node is a leaf, this stores the output label
        self.leaf_label = ""

        # is true iff this node is a leaf node
        self.is_leaf_node = is_leaf_node
        if is_leaf_node:
            self.leaf_label = label
        else:
            self.attribute = attribute_int

    # adds a new map for next_nodes
    def add_mapping(self, value, node):
        self.next_nodes[value] = node

    # Runs the decision tree on the given input
    def evaluate(self, input_data):
        if self.is_leaf_node:
            return self.leaf_label
        return self.next_nodes[input_data[self.attribute]].evaluate(input_data)


# Essentially a wrapper class for the root node,
# to add some privacy to the node class. In particular,
# the behavior of the tree is reliant on the fact that
# the nodes are well formed, so we don't want the
# user to be able to mess with it.
class DecisionTree:

    def __init__(self, root):
        self.root = root

    # Returns the decision trees labeling run on
    # the input data. Note that the input_data
    # must have the data in the same order as the training data.
    def evaluate(self, input_data):
        return self.root.evaluate(input_data)
