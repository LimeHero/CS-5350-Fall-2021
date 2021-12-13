import random
import numpy as np
import nn_sgd
import neural_network


def main():
    # forward_pass_test()
    # paper_problem_test()
    bank_note_test()


def identity(x):
    return x


def forward_pass_test():
    nn = neural_network.NeuralNetwork(3, 2, 2, identity)
    nn.weights[0][0][0] = -1.
    nn.weights[0][1][0] = 2.
    nn.weights[0][2][0] = -1.

    nn.weights[0][0][1] = 2.
    nn.weights[0][1][1] = 1.
    nn.weights[0][2][1] = -1.

    nn.weights[1][0][0] = -1.
    nn.weights[1][1][0] = 2.
    nn.weights[1][2][0] = 3.

    nn.weights[1][0][1] = 1.
    nn.weights[1][1][1] = -1.
    nn.weights[1][2][1] = 0.

    nn.weights[2][0][0] = 2
    nn.weights[2][1][0] = 1
    nn.weights[2][2][0] = -1

    print(nn.evaluate([1, 1]))


def paper_problem_test():
    values = [[1, 1]]
    labels = [1]
    nn = nn_sgd.nn_sgd(values, labels, 1, [3, 2], 1)


def bank_note_test():
    train_values = []
    train_labels = []
    with open("Data/bank_note/train.csv", 'r') as f:
        for line in f:
            terms = line.split(",")
            train_values.append([])
            for i in range(len(terms) - 1):
                train_values[-1].append(float(terms[i]))
            train_labels.append(2 * float(terms[-1]) - 1.)  # so the labels are +- 1

    test_values = []
    test_labels = []
    with open("Data/bank_note/test.csv", 'r') as f:
        for line in f:
            terms = line.split(",")
            test_values.append([])
            for i in range(len(terms) - 1):
                test_values[-1].append(float(terms[i]))
            test_labels.append(2 * float(terms[-1]) - 1.)  # so the labels are +- 1

    d = 20.

    # desired learning schedule
    def learning_schedule(initial, t):
        return initial / (1 + (initial / d) * t)

    layers = 3
    for width in [5, 10, 25, 50, 100]:
        convergence = .2
        initial_learn = .5
        nn = nn_sgd.nn_sgd(train_values, train_labels, convergence, [layers, width], initial_learn, print_info=False)

        incorrect_train_label_count = 0
        for i in range(len(train_labels)):
            incorrect_train_label_count += nn.evaluate(train_values[i]) * train_labels[i] < 0

        incorrect_label_count = 0
        for i in range(len(test_labels)):
            incorrect_label_count += nn.evaluate(test_values[i]) * test_labels[i] < 0

        print("width: " + str(width))
        print("incorrect_train_label_count: " + str(incorrect_train_label_count))
        print("incorrect_label_count: " + str(incorrect_label_count))
        print("average train error: " + str(incorrect_train_label_count / len(train_labels)))
        print("average test error: " + str(incorrect_label_count / len(test_labels)))
        print()
        print()


if __name__ == "__main__":
    main()
