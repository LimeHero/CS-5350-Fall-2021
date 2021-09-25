import tree
import id3


def main():
    car_test()


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

    my_id3_tree = id3.categorical_id3(x_values, x_labels, "info gain", -1)

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
    tennis_decision_tree = id3.categorical_id3(tennis_values, tennis_labels, "maj error", -1)
    print(tennis_decision_tree.evaluate(["R", "H", "H", "S"]))


# to run the experiments for the assignment
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

    gain_functions = ["gini", "info gain", "maj error"]
    print("on test data:")
    for max_depth in range(1, 7):
        for gain_function in gain_functions:
            id3_tree = id3.categorical_id3(train_values, train_labels, gain_function, max_depth)

            correct_num = 0
            incorrect_num = 0
            for i in range(len(test_labels)):
                if id3_tree.evaluate(test_values[i]) == test_labels[i]:
                    correct_num += 1
                else:
                    incorrect_num += 1

            print("max depth: " + str(max_depth))
            print("gain function: " + gain_function)
            print("correct number: " + str(correct_num))
            print("incorrect number: " + str(incorrect_num))
            print("accuracy: " + str(correct_num / (correct_num + incorrect_num)))

    print()
    print("on training data:")
    for max_depth in range(1, 7):
        for gain_function in gain_functions:
            id3_tree = id3.categorical_id3(train_values, train_labels, gain_function, max_depth)

            correct_num = 0
            incorrect_num = 0
            for i in range(len(train_labels)):
                if id3_tree.evaluate(train_values[i]) == train_labels[i]:
                    correct_num += 1
                else:
                    incorrect_num += 1

            print("max depth: " + str(max_depth))
            print("gain function: " + gain_function)
            print("correct number: " + str(correct_num))
            print("incorrect number: " + str(incorrect_num))
            print("accuracy: " + str(correct_num / (correct_num + incorrect_num)))

def test_results()

if __name__ == "__main__":
    main()
