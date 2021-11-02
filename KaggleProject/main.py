import math

import DecisionTree.tree
import DecisionTree.id3
import DecisionTree.bagged_tree as bagged_tree

import EnsembleLearning.ada_boost

# this file is used to run experiments for the final and midterm projects
# in CS 5350.
# In general, there will be helper functions that are "tests"
# in the sense that they divide the test data into two groups and measure
# the effectiveness of the algorithm on the test data.
#
# Then, there will be corresponding functions that use the same algorithm
# to produce an output for the actual test data.


test_train_ratio = .9  # the number of test examples used divided by total examples


def main():
    bagged_tree_output()


# this is a function that runs a test using adaboost on the training data
def adaboost_test():
    data = process_test_data()
    train_values = data[0]
    train_labels = data[1]
    test_values = data[2]
    test_labels = data[3]

    ada_boost_result = EnsembleLearning.ada_boost.ada_boost(train_values, train_labels, "info gain", 500,
                                                            print_info=True, test_values=test_values,
                                                            test_labels=test_labels)


# this prints the actual test results using adaboost on the test data
def adaboost_output():
    data = process_output_data()
    train_values = data[0]
    train_labels = data[1]
    test_values = data[2]
    print(len(test_values))

    ada_boost_result = EnsembleLearning.ada_boost.ada_boost(train_values, train_labels, "info gain", 20)

    print("ID,Prediction")
    for i in range(len(test_values)):
        ith_result = ada_boost_result.evaluate(test_values[i], print_avg=True)
        print(str(i + 1) + "," + str(ith_result))


def bagged_tree_test():
    data = process_test_data()
    train_values = data[0]
    train_labels = data[1]
    test_values = data[2]
    test_labels = data[3]

    all_attributes = get_all_attributes(train_values, test_values)

    bagged_result = bagged_tree.BaggedTree()
    bagged_result.initialize_id3(train_values, train_labels, 100, "info gain", len(train_values), print_info=True,
                                 test_values=test_values, test_labels=test_labels, attr_subset_size=6,
                                 all_attributes=all_attributes)


def bagged_tree_output():
    data = process_output_data()
    train_values = data[0]
    train_labels = data[1]
    test_values = data[2]

    all_attributes = get_all_attributes(train_values, test_values)

    bagged_result = bagged_tree.BaggedTree()
    bagged_result.initialize_id3(train_values, train_labels, 400, "info gain", len(train_values), attr_subset_size=6,
                                 all_attributes=all_attributes)

    print("ID,Prediction")

    for i in range(len(test_values)):
        ith_result = bagged_result.evaluate(test_values[i], avg=True)
        print(str(i + 1) + "," + str(ith_result))


def get_all_attributes(train_values, test_values):
    all_attributes = []
    for j in range(len(train_values[0])):
        all_attributes.append(set())

    for j in range(len(train_values[0])):
        if type(train_values[0][j]) != str:
            all_attributes[j].add("less")
            all_attributes[j].add("more")
            continue

        for i in range(len(train_values)):
            all_attributes[j].add(train_values[i][j])
        for i in range(len(test_values)):
            all_attributes[j].add(test_values[i][j])

    return all_attributes


# returns [train_values, train_labels, test_values, test_labels]
def process_test_data():
    train_values = []
    train_labels = []

    test_values = []
    test_labels = []
    test_count = -1
    with open("Data/train_final.csv", 'r') as f:
        for line in f:
            if test_count == -1:
                test_count = 0
                continue
            test_count += 1 - test_train_ratio

            # add to to the testing set
            if test_count > 1:
                test_count -= 1
                terms = line.strip().split(',')
                test_values.append([])
                for i in range(len(terms) - 1):
                    try:
                        test_values[-1].append(float(terms[i]))
                    except ValueError:
                        test_values[-1].append(terms[i])
                test_labels.append(float(terms[-1]))
            # add to the training set
            else:
                terms = line.strip().split(',')
                train_values.append([])
                for i in range(len(terms) - 1):
                    try:
                        train_values[-1].append(float(terms[i]))
                    except ValueError:
                        train_values[-1].append(terms[i])
                train_labels.append(float(terms[-1]))

    return [train_values, train_labels, test_values, test_labels]


def process_output_data():
    train_values = []
    train_labels = []

    first_loop = True
    with open("Data/train_final.csv", 'r') as f:
        for line in f:
            if first_loop:
                first_loop = False
                continue

            terms = line.strip().split(',')
            train_values.append([])
            for i in range(len(terms) - 1):
                try:
                    train_values[-1].append(float(terms[i]))
                except ValueError:
                    train_values[-1].append(terms[i])
            train_labels.append(float(terms[-1]))

    test_values = []
    first_loop = True
    with open("Data/test_final.csv", 'r') as f:
        for line in f:
            if first_loop:
                first_loop = False
                continue

            terms = line.strip().split(',')
            test_values.append([])
            for i in range(1, len(terms)):
                try:
                    test_values[-1].append(float(terms[i]))
                except ValueError:
                    test_values[-1].append(terms[i])

    return [train_values, train_labels, test_values]


# a function meant to move values in [0,1]
# closer to the endpoints, so fewer guesses
# in the range (.25, .75)
def strengthen_prediction(x):
    if x > .5:
        return math.sqrt(2*x - 1)/2 + .5

    return .5 - math.sqrt(1 - 2*x)/2


if __name__ == "__main__":
    main()
