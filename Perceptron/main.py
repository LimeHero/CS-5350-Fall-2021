import perceptron


def main():
    print("vote perceptron test: ")
    bank_note_test("vote")
    print()
    print()
    print("standard perceptron test:")
    bank_note_test("standard")
    print()
    print()
    print("average perceptron test:")
    bank_note_test("average")


def disjunction_test(n, k):
    values = []
    labels = []
    next_value = [0] * n
    while next_value[0] < 2:
        values.append(next_value.copy())
        for i in range(k):
            if next_value[i] == 0:
                labels.append(1)
                break

        if len(labels) < len(values):
            for i in range(k):
                if next_value[k + i] == 1:
                    labels.append(1)
                    break

        if len(labels) < len(values):
            labels.append(-1)

        next_value[-1] += 1
        for i in range(len(next_value) - 1):
            if next_value[len(next_value) - i - 1] > 1:
                next_value[len(next_value) - i - 1] = 0
                next_value[len(next_value) - i - 2] += 1

    perceptron.perceptron(values, labels, 1000, .1, perc_type="average")


# runs the tests for the bank_note experiments
# required for homework 3
def bank_note_test(perc_type):
    train_values = []
    train_labels = []
    with open("Data/bank_note/train.csv", 'r') as f:
        for line in f:
            terms = line.split(",")
            train_values.append([])
            for i in range(len(terms) - 1):
                train_values[-1].append(float(terms[i]))
            train_labels.append(2 * float(terms[-1]) - 1)  # so the labels are +- 1

    test_values = []
    test_labels = []
    with open("Data/bank_note/test.csv", 'r') as f:
        for line in f:
            terms = line.split(",")
            test_values.append([])
            for i in range(len(terms) - 1):
                test_values[-1].append(float(terms[i]))
            test_labels.append(2 * float(terms[-1]) - 1.)  # so the labels are +- 1

    perc_result = perceptron.perceptron(test_values, test_labels, 10, .01, perc_type=perc_type, print_info=True)

    incorrect_label_count = 0
    for i in range(len(test_labels)):
        if perc_result.evaluate(test_values[i]) != test_labels[i]:
            incorrect_label_count += 1

    print("average test error " + str(incorrect_label_count / len(test_labels)))


if __name__ == "__main__":
    main()
