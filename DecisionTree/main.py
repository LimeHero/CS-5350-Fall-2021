import id3


def main():
    car_test()
    bank_test()


# this test passed: id3 worked as in the handwritten example
def x_test():
    x_values = [["0", "0", "1", "0"],
                ["0", "1", "0", "0"],
                ["0", "0", "1", "1"],
                ["1", "0", "0", "1"],
                ["0", "1", "1", "0"],
                ["1", "1", "0", "0"],
                ["0", "1", "0", "1"],
                ]

    x_labels = ["0", "0", "1", "1", "0", "0", "0"]

    my_id3_tree = id3.id3(x_values, x_labels, "info gain", -1)

    test = ["0", "0", "1", "0"]  # 0
    print(my_id3_tree.evaluate(test))
    test = ["0", "0", "1", "1"]  # 1
    print(my_id3_tree.evaluate(test))
    test = ["0", "1", "0", "1"]  # 0
    print(my_id3_tree.evaluate(test))
    test = ["0", "0", "0", "1"]  # 1
    print(my_id3_tree.evaluate(test))
    test = ["1", "0", "1", "1"]  # 1
    print(my_id3_tree.evaluate(test))
    test = ["1", "1", "1", "0"]  # 0
    print(my_id3_tree.evaluate(test))
    test = ["1", "0", "1", "0"]  # 0
    print(my_id3_tree.evaluate(test))


# this test passed: id3 worked as in the handwritten example
def tennis_test():
    tennis_values = [["S", "H", "H", "W"],
                     ["S", "H", "H", "S"],
                     ["O", "H", "H", "W"],
                     ["R", "M", "H", "W"],
                     ["R", "C", "N", "W"],
                     ["R", "C", "N", "S"],
                     ["O", "C", "N", "S"],
                     ["S", "M", "H", "W"],
                     ["S", "C", "N", "W"],
                     ["R", "M", "N", "W"],
                     ["S", "M", "N", "S"],
                     ["O", "M", "H", "S"],
                     ["O", "H", "N", "W"],
                     ["R", "M", "H", "S"],
                     ]

    tennis_labels = ["-",
                     "-",
                     "+",
                     "+",
                     "+",
                     "-",
                     "+",
                     "-",
                     "+",
                     "+",
                     "+",
                     "+",
                     "+",
                     "-",
                     ]

    # tennis set worked with info gain :)
    tennis_decision_tree = id3.id3(tennis_values, tennis_labels, "maj error", -1)
    print(tennis_decision_tree.evaluate(["R", "H", "H", "S"]))


# to run the experiments for the assignment, question 1
def car_test():
    train_values = []
    train_labels = []
    with open("Data/car/train.csv", 'r') as f:
        for line in f:
            terms = line.strip().split(',')
            train_values.append(terms[0:(len(terms) - 1)])
            train_labels.append(terms[-1])

    test_values = []
    test_labels = []
    with open("Data/car/test.csv", 'r') as f:
        for line in f:
            terms = line.strip().split(',')
            test_values.append(terms[0:(len(terms) - 1)])
            test_labels.append(terms[-1])

    print("car on test data:")
    test_results(train_values, train_labels, test_values, test_labels, 6)
    print()
    print()
    print()
    print("car on training data:")
    test_results(train_values, train_labels, train_values, train_labels, 6)
    print()
    print()
    print()


# to run the experiments for the assignment, question 2
def bank_test():
    train_values = []
    train_labels = []
    with open("Data/bank/train.csv", 'r') as f:
        for line in f:
            terms = line.strip().split(',')
            train_values.append(terms[0:(len(terms) - 1)])
            train_labels.append(terms[-1])
    test_values = []
    test_labels = []
    with open("Data/bank/test.csv", 'r') as f:
        for line in f:
            terms = line.strip().split(',')
            test_values.append(terms[0:(len(terms) - 1)])
            test_labels.append(terms[-1])

    process_bank_attributes(train_values, test_values)

    process_unknowns(train_values, test_values)

    print("bank on test data:")
    test_results(train_values, train_labels, test_values, test_labels, 16)
    print()
    print()
    print()
    print("bank on training data:")
    test_results(train_values, train_labels, train_values, train_labels, 16)
    print()
    print()
    print()


# categorizes the bank values arrays.
# extremely specific to this data set
def process_bank_attributes(train_values, test_values):
    numeric_attributes = [0, 5, 9, 11, 12, 14] # 13 is handled separately
    for num_attr in numeric_attributes:
        nth_attribute = []
        for i in range(len(train_values)):
            train_values[i][num_attr] = int(train_values[i][num_attr])
            nth_attribute.append(train_values[i][num_attr])
        nth_attribute.sort()
        median = nth_attribute[int(len(nth_attribute)/2)]

        for i in range(len(train_values)):
            if train_values[i][num_attr] <= median:
                train_values[i][num_attr] = "less"  # less than median
            else:
                train_values[i][num_attr] = "more"  # more than median

        for i in range(len(test_values)):
            if int(test_values[i][num_attr]) <= median:
                test_values[i][num_attr] = "less"  # less than median
            else:
                test_values[i][num_attr] = "more"  # more than median

    # 13th attribute case
    nth_attribute = []
    for i in range(len(train_values)):
        train_values[i][13] = int(train_values[i][13])
        if train_values[i][13] >= 0:
            nth_attribute.append(train_values[i][13])
    nth_attribute.sort()
    median = nth_attribute[int(len(nth_attribute)/2)]

    for i in range(len(train_values)):
        if train_values[i][13] < 0:
            train_values[i][13] = "neither"
            continue

        if train_values[i][13] <= median:
            train_values[i][13] = "less"  # less than median
        else:
            train_values[i][13] = "more"  # more than median

    for i in range(len(test_values)):
        if int(test_values[i][13]) < 0:
            test_values[i][13] = "neither"
            continue

        if int(test_values[i][13]) <= median:
            test_values[i][13] = "less"  # less than median
        else:
            test_values[i][13] = "more"  # more than median


# changes unknowns to the most common attribute value
def process_unknowns(train_values, test_values):
    for attr in range(len(train_values[0])):
        # the frequency of each value in the training set
        counts = {}
        for i in range(len(train_values)):
            if train_values[i][attr] == "unknown":
                continue

            if not (train_values[i][attr] in counts):
                counts[train_values[i][attr]] = 0

            counts[train_values[i][attr]] += 1

        most_freq_attr = ""
        counts[""] = -1
        for value in counts.keys():
            if counts[value] > counts[most_freq_attr]:
                most_freq_attr = value

        for i in range(len(train_values)):
            if train_values[i][attr] == "unknown":
                train_values[i][attr] = most_freq_attr

        for i in range(len(test_values)):
            if test_values[i][attr] == "unknown":
                test_values[i][attr] = most_freq_attr


# prints the test results for the given training and testing data
# according to the assignments requirements
def test_results(train_values, train_labels, test_values, test_labels, max_max_depth):
    gain_functions = ["gini", "info gain", "maj error"]
    for gain_function in gain_functions:
        accuracy_mean = 0
        for max_depth in range(1, max_max_depth+1):
            id3_tree = id3.id3(train_values, train_labels, gain_function, max_depth)

            correct_num = 0
            incorrect_num = 0
            for i in range(len(test_labels)):
                if id3_tree.evaluate(test_values[i]) == test_labels[i]:
                    correct_num += 1
                else:
                    incorrect_num += 1

            accuracy = correct_num / (correct_num + incorrect_num)
            accuracy_mean += accuracy
            # print("max depth: " + str(max_depth))
            # print("gain function: " + gain_function)
            # print("correct number: " + str(correct_num))
            # print("incorrect number: " + str(incorrect_num))
            # print("accuracy: " + str(accuracy))
            # print()
        print("average accuracy: " + str(accuracy_mean/max_max_depth))
        print("with " + gain_function)


if __name__ == "__main__":
    main()
