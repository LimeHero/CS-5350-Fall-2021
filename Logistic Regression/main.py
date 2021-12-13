import sgd


def main():
    print("logistic regression main!")
    bank_note_test()


# logistic regression tests for hw 5s
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

    def learning_schedule(initial, t):
        return initial / (1 + (initial / d) * t)

    num_epochs = 100
    initial_learn = .001
    for map in [False, True]:
        for variance in [.01, .1, .5, 1, 3, 5, 10, 100]:
            sgd_result = sgd.lin_regression_sgd(train_values, train_labels, num_epochs, variance, initial_learn,
                                                learning_schedule=learning_schedule, print_info=False, map=map)

            incorrect_label_count = 0
            for i in range(len(test_labels)):
                if sgd_result.evaluate(test_values[i]) != test_labels[i]:
                    incorrect_label_count += 1

            incorrect_train_label_count = 0
            for i in range(len(train_labels)):
                if sgd_result.evaluate(train_values[i]) != train_labels[i]:
                    incorrect_train_label_count += 1

            print("map = " + str(map))
            print("variance = " + str(variance))
            print("average train error: " + str(incorrect_train_label_count / len(train_labels)))
            print("average test error: " + str(incorrect_label_count / len(test_labels)))
            print()
            print()


if __name__ == "__main__":
    main()
