import ada_boost


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

    stump_forest = ada_boost.ada_boost(x_values, x_labels, "info gain", 1)
    for i in range(len(x_values)):
        print(stump_forest.evaluate(x_values[i]))
    print("Ensemble Learning Test:")
    try:
        float(str)
        numeric = True
    except ValueError:
        numeric = False


if __name__ == "__main__":
    main()
