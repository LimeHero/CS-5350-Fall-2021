import SVM
import numpy as np


def main():
    print("SVM main function.")
    # np_testing()
    bank_note_test_sgd()
    bank_note_test_dot(True)


# to help my understanding of np functions
def np_testing():
    myw = np.array([0.] * 4)

    myw += 4. * 2. * np.array([.5, .2, .1, .2])
    print(myw)
    values = [[1, 0], [0, 1], [1, 1]]
    labels = [1, -1, 1]

    xy_xyt = np.array([np.dot(xi, xj) for xi in values for xj in values])
    print(xy_xyt)
    xy_xyt = xy_xyt.reshape((len(values), len(values)))
    print(xy_xyt)

    xy_xyt = (np.array(labels) * (xy_xyt * np.array(labels)).T)
    print(xy_xyt)

    alphas = np.array([1, 2, 3])
    xy_xyt = (np.array(alphas) * (xy_xyt * np.array(alphas)).T)
    print(xy_xyt)

    print(xy_xyt.sum())


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

    for learn_sched in range(1):
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

            print("vector values: " + str(svm_result.vector))
            print("c = " + str(c))
            print("average train error: " + str(incorrect_train_label_count / len(train_labels)))
            print("average test error: " + str(incorrect_label_count / len(test_labels)))
            print()
            print()


def bank_note_test_dot(use_gaussian):
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

    prev_supp_vectors = None
    for gamma in [0.1, 0.5, 1., 5., 100.]:
        for c in [100 / 873, 500 / 873, 700 / 873]:

            # takes in two numpy arrays and returns their gaussian product
            def gaussian_kernel(x, y):
                return np.exp(- np.dot(x - y, x - y) / gamma)

            if use_gaussian:
                svm_result = SVM.svm(train_values, train_labels, c, kernel=gaussian_kernel)
            else:
                svm_result = SVM.svm(train_values, train_labels, c)

            incorrect_label_count = 0
            for i in range(len(test_labels)):
                if svm_result.evaluate(test_values[i]) != test_labels[i]:
                    incorrect_label_count += 1

            incorrect_train_label_count = 0
            for i in range(len(train_labels)):
                if svm_result.evaluate(train_values[i]) != train_labels[i]:
                    incorrect_train_label_count += 1

            if prev_supp_vectors is not None and -.1 < c - 500/873 < .1:
                count = 0
                for i in range(len(prev_supp_vectors)):
                    if prev_supp_vectors[i] in svm_result.vector:
                        count += 1
                print("num same supp vectors: " + str(count))
            print("gamma: " + str(gamma))
            print("num supp vectors: " + str(len(svm_result.vector)))
            print("c = " + str(c))
            print("average train error: " + str(incorrect_train_label_count / len(train_labels)))
            print("average test error: " + str(incorrect_label_count / len(test_labels)))
            print()
            print()
            if -.1 < c - 500/873 < .1:
                prev_supp_vectors = svm_result.vector


if __name__ == "__main__":
    main()
