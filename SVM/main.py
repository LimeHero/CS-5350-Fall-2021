import SVM
import numpy as np


def main():
    print("SVM main function.")
    bank_note_test_dual()


def sgd_test():
    values = [[.5, -1, .3],
              [-1, -2, -2],
              [1.5, .2, -2.5]]

    labels = [1, -1, 1]

    def learning_schedule(initial, t):
        return initial / (5 ** t)

    SVM.svm_sgd(values, labels, 1, 1, .01, learning_schedule=learning_schedule, print_info=True)


# runs the tests for the bank_note experiments
# required for homework 3
def bank_note_test_sgd():
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

    a = 1

    def learning_schedule(initial, t):
        return initial / (1 + (a / initial) * t)

    for learn_sched in range(2):
        for c in [100 / 873, 500 / 873, 700 / 873]:
            if learn_sched == 0:
                svm_result = SVM.svm_sgd(train_values, train_labels, 100, c, 5, learning_schedule=learning_schedule)
            else:
                svm_result = SVM.svm_sgd(train_values, train_labels, 100, c, 5)

            incorrect_label_count = 0
            for i in range(len(test_labels)):
                if svm_result.evaluate(test_values[i]) != test_labels[i]:
                    incorrect_label_count += 1

            incorrect_train_label_count = 0
            for i in range(len(train_labels)):
                if svm_result.evaluate(train_values[i]) != train_labels[i]:
                    incorrect_train_label_count += 1

            if learn_sched == 0:
                print("learning schedule: r_0/(1+(r_0 / a)*t)")
            else:
                print("linear learning schedule")

            print("c = " + str(c))
            print("average train error: " + str(incorrect_train_label_count / len(train_labels)))
            print("average test error: " + str(incorrect_label_count / len(test_labels)))
            print()
            print()


def bank_note_test_dual():
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

    for c in [100 / 873, 500 / 873, 700 / 873]:

        gamma = 1

        # takes in two numpy arrays and returns their gaussian product
        def gaussian_kernel(x, y):
            np.exp(- np.dot(x, y) / gamma)

        svm_result = SVM.svm(train_values, train_labels, c)

        incorrect_label_count = 0
        for i in range(len(test_labels)):
            if svm_result.evaluate(test_values[i]) != test_labels[i]:
                incorrect_label_count += 1

        incorrect_train_label_count = 0
        for i in range(len(train_labels)):
            if svm_result.evaluate(train_values[i]) != train_labels[i]:
                incorrect_train_label_count += 1

        print("c = " + str(c))
        print("average train error: " + str(incorrect_train_label_count / len(train_labels)))
        print("average test error: " + str(incorrect_label_count / len(test_labels)))


if __name__ == "__main__":
    main()
