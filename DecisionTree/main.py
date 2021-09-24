import tree


def main():
    print("hello!")
    my_root = tree.DecisionNode(False, 0, "")
    a = tree.DecisionNode(False, 1, "")
    c = tree.DecisionNode(False, 2, "")
    a.add_mapping("x", tree.DecisionNode(True, 0, "+"))
    a.add_mapping("y", tree.DecisionNode(True, 0, "-"))
    c.add_mapping("bla", tree.DecisionNode(True, 0, "+..."))
    c.add_mapping("bla1", tree.DecisionNode(True, 0, "-..."))
    my_root.add_mapping("a", a)
    my_root.add_mapping("c", c)
    my_root.add_mapping("b", tree.DecisionNode(True, 0, "++"))

    my_tree = tree.DecisionTree(my_root)

    my_data = ["c", "x", "bla"]

    print(my_tree.evaluate(my_data))


if __name__ == "__main__":
    main()
