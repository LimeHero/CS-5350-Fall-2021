import lms

def main():
    train_values = []
    train_labels = []
    with open("Data/SLUMP/train.csv", 'r') as f:
        for line in f:
            terms = line.strip().split(',')
            for i in range(len(terms)-1):
                train_values.append(float(terms[i]))
            train_labels.append(float(terms[-1]))

    test_values = []
    test_labels = []
    with open("Data/SLUMP/test.csv", 'r') as f:
        for line in f:
            terms = line.strip().split(',')
            for i in range(len(terms)-1):
                test_values.append(float(terms[i]))
            test_labels.append(float(terms[-1]))

    lms.lms(train_values, train_labels, .5, descent_type="grad", print_info=True,
            test_values=test_values, test_labels=test_labels)


if __name__ == "main":
    main()
