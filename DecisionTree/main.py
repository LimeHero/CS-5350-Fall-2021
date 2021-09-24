import tree
import id3


def main():
    print("hello!")
    my_root = tree.DecisionNode(False, 0, "")
    a = tree.DecisionNode(False, 1, "")
    c = tree.DecisionNode(False, 2, "")
    a.add_mapping("x", tree.DecisionNode(True, 0, "+"))
    a.add_mapping("y", tree.DecisionNode(True, 0, "-"))
    c.add_mapping("x", tree.DecisionNode(True, 0, "+..."))
    c.add_mapping("bla", tree.DecisionNode(True, 0, "-..."))
    my_root.add_mapping("a", a)
    my_root.add_mapping("c", c)
    my_root.add_mapping("b", tree.DecisionNode(True, 0, "++"))

    my_tree = tree.DecisionTree(my_root)

    print(my_tree.evaluate(["c", "x", "x"]))
    print(my_tree.evaluate(["b", "x", "bla"]))
    print(my_tree.evaluate(["a", "y", "bla"]))
    print(my_tree.evaluate(["c", "y", "bla"]))
    print()
    print()

    id3_test_values = [["0", "0", "1", "0"],
                       ["0", "1", "0", "0"],
                       ["0", "0", "1", "1"],
                       ["1", "0", "0", "1"],
                       ["0", "1", "1", "0"],
                       ["1", "1", "0", "0"],
                       ["0", "1", "0", "1"],
                       ]

    id3_test_labels = ["0", "0", "1", "1", "0", "0", "0"]

    my_id3_tree = id3.categorical_id3(id3_test_values, id3_test_labels, "info gain", -1)

    test = ["0", "0", "1", "0"] # 0
    print(my_id3_tree.evaluate(test))
    test = ["0", "0", "1", "1"] # 1
    print(my_id3_tree.evaluate(test))
    test = ["0", "1", "0", "1"] # 0
    print(my_id3_tree.evaluate(test))
    test = ["0", "0", "0", "1"] # 1
    print(my_id3_tree.evaluate(test))
    test = ["1", "0", "1", "1"] # 1
    print(my_id3_tree.evaluate(test))
    test = ["1", "1", "1", "0"] # 0
    print(my_id3_tree.evaluate(test))
    test = ["1", "0", "1", "0"] # 0
    print(my_id3_tree.evaluate(test))


if __name__ == "__main__":
    main()
