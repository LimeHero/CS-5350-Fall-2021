import random
import tree
import id3


# a simple implementation of a bagged tree
class BaggedTree:

    # empty constructor
    def __init__(self):
        self.trees = []

    # runs id3 repeatedly to form this bagged tree
    def initialize_id3(self, train_values, train_labels, gens, gain_function, m, print_info=False,
                       test_values=None, test_labels=None, attr_subset_size=-1, all_attributes=None):
        for gen in range(gens):
            new_value_set = []
            new_label_set = []
            for i in range(m):
                index = random.randrange(0, len(train_values))
                new_value_set.append(train_values[index])
                new_label_set.append(train_labels[index])

            new_tree = id3.id3(new_value_set, new_label_set, gain_function, -1, attr_subset_size=attr_subset_size,
                               all_attributes=all_attributes)

            self.trees.append(new_tree)

            if print_info:
                print(gen, end=",")

                error_count = 0
                for i in range(len(train_labels)):
                    if self.evaluate(train_values[i]) != train_labels[i]:
                        error_count += 1

                print(error_count / len(train_labels), end=",")

                error_count = 0
                for i in range(len(test_labels)):
                    if self.evaluate(test_values[i]) != test_labels[i]:
                        error_count += 1

                print(error_count / len(test_labels))

    def evaluate(self, values):
        label_counts = {}
        for j in range(len(self.trees)):
            label = self.trees[j].evaluate(values)
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1

        max_label = ""
        max_label_freq = -1

        for label in label_counts.keys():
            if label_counts[label] > max_label_freq:
                max_label = label
                max_label_freq = label_counts[label]

        return max_label
