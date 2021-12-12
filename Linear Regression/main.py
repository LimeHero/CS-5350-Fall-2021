import lms
import numpy as np


def main():
    print("gradient test:")
    lms_test("grad")
    print()
    print()
    print()
    print("sgd test:")
    lms_test("sgd")


def lms_test(descent_type):

    train_values = []
    train_labels = []
    with open("Data/SLUMP/train.csv", 'r') as f:
        for line in f:
            terms = line.strip().split(',')
            train_values.append([])
            for i in range(len(terms) - 1):
                train_values[-1].append(float(terms[i]))
            train_labels.append(float(terms[-1]))

    test_values = []
    test_labels = []
    with open("Data/SLUMP/test.csv", 'r') as f:
        for line in f:
            terms = line.strip().split(',')
            test_values.append([])
            for i in range(len(terms) - 1):
                test_values[-1].append(float(terms[i]))
            test_labels.append(float(terms[-1]))

    grad_result = lms.lms(train_values, train_labels, .005, descent_type=descent_type, print_info=True)

    print("result:")
    print(grad_result)

    print("test cost:")
    cost = 0
    for j in range(len(test_values)):
        dot_prod = 0
        for i in range(len(test_values[j])):
            dot_prod += grad_result[i] * test_values[j][i]
        dot_prod += grad_result[-1]
    cost += (test_labels[j] - dot_prod) ** 2

    print(cost)

    for i in range(len(train_values)):
        train_values[i].append(1)

    # m*m^T
    mmT = []
    for i in range(len(train_values[0])):
        mmT.append([])
        for j in range(len(train_values[0])):
            mmT[-1].append(0)

    # for i in range(len(train_values[0])):
    #    for j in range(len(train_values[0])):
    #        a_ij = 0
    #        for k in range(len(train_values)):
    #            a_ij += train_values[k][i]*train_values[k][j]
    #        mmT[i][j] = a_ij

    # for i in range(len(mmT)):
    #    print(end="{")
    #    for j in range(len(mmT)-1):
    #        print(str(mmT[i][j]) + ",", end = "")
    #    print(str(mmT[i][-1])+"},", end="")
    # print()
    # print(test_labels)
    # print(np.matmul(train_labels, np.matmul(train_values, np.linalg.inv(mmT))))


if __name__ == "__main__":
    main()
