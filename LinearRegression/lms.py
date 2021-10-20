# attempts to minimize the cost of w*values,
# where w is the desired vector we return
#
# r is the learning rate.
#
# descent_type must be either "sgd" or "grad",
# which stands for (regular) gradient descent
def lms(values, labels, r, descent_type="sgd", print_info=False, test_values=None, test_labels=None):
    if descent_type != "sgd" and descent_type != "grad":
        raise Exception("descent_type must be equal to \"sgd\" or \"grad\"")

    w = [0]*len(values[0])
    # value of w in the previous step to check for convergence
    w_ = [1]*len(values[0])

    # the index of the next example, if using sgd
    index = -1
    while True:
        index += 1
        if index == len(labels):
            index = 0

        diff = 0
        for i in range(len(w)):
            diff += (w[i] - w_[i]) ** 2

        if diff < .00001:
            break

        w_ = w

        gradient = [0]*len(w)
        if descent_type == "sgd":
            gradient = lms_gradient(w, values[index], labels[index])

        if descent_type == "grad":
            for i in range(len(labels)):
                next_grad = lms_gradient(w, values[i], labels[i])
                for j in range(len(w)):
                    gradient[j] += next_grad[j]

        for i in range(len(w)):
            w[i] += gradient * r


# helper function to find the gradient for a single
# element, which is how sgd is defined, or the total
# gradient is the sum
def lms_gradient(w, value, label):
    grad = []
    for i in range(len(value)):
        grad.append(-(label - w[i]*value[i])*value[i])

    return grad





