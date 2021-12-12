import random
import numpy as np
import nn_sgd


def main():
    # paper_problem_test()
    bank_note_test()


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

    d = 10

    # desired learning schedule
    def learning_schedule(initial, t):
        return initial / (1 + (initial / d) * t)

    layers = 3
    for width in [5]:
        num_epochs = 10
        initial_learn = 50
        nn = nn_sgd.nn_sgd(train_values, train_labels, num_epochs, [layers, width], initial_learn, print_info=True)

        incorrect_label_count = 0
        for i in range(len(test_labels)):
            if nn.evaluate(test_values[i]) != test_labels[i]:
                incorrect_label_count += 1

        incorrect_train_label_count = 0
        for i in range(len(train_labels)):
            if nn.evaluate(train_values[i]) != train_labels[i]:
                incorrect_train_label_count += 1

        print("width: " + str(width))
        print("incorrect_train_label_count: " + str(incorrect_label_count))
        print("incorrect_label_count: " + str(incorrect_label_count))
        print("average train error: " + str(incorrect_train_label_count / len(train_labels)))
        print("average test error: " + str(incorrect_label_count / len(test_labels)))
        print()
        print()


if __name__ == "__main__":
    main()
