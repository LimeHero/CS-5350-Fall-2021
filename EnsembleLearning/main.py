import ada_boost
# import sys
# sys.path.insert(0, '/CS-5350-Fall-2021/DecisionTree/')
# from DecisionTree import tree
# from DecisionTree import id3


def main():
    x_values = [["0", "0", "1", "0"],
                ["0", "1", "0", "0"],
                ["0", "0", "1", "1"],
                ["1", "0", "0", "1"],
                ["0", "1", "1", "0"],
                ["1", "1", "0", "0"],
                ["0", "1", "0", "1"],
                ]

    x_labels = ["0", "0", "1", "1", "0", "0", "0"]

    # stump_forest = ada_boost.ada_boost(x_values, x_labels, "info gain", 50, print_info=False, test_values=x_values,
    #                                    test_labels=x_labels)
    print("EnsembleLearning Test:")
    bank_test()


# to run the experiments for the assignment, question 2 part a
def bank_test():
    train_values = []
    train_labels = []
    with open("Data/bank/train.csv", 'r') as f:
        for line in f:
            terms = line.strip().split(',')
            train_values.append([])
            for j in range(len(terms)-1):
                try:
                    train_values[-1].append(float(terms[j]))
                except ValueError:
                    train_values[-1].append(terms[j])
            train_labels.append(terms[-1])

    test_values = []
    test_labels = []
    with open("Data/bank/test.csv", 'r') as f:
        for line in f:
            terms = line.strip().split(',')
            test_values.append([])
            for j in range(len(terms)):
                try:
                    test_values[-1].append(float(terms[j]))
                except ValueError:
                    test_values[-1].append(terms[j])
            test_labels.append(terms[-1])

    ada_boost_result = ada_boost.ada_boost(train_values, train_labels, "info gain", 500, print_info=True,
                                           test_values=test_values, test_labels=test_labels)



if __name__ == "__main__":
    main()
