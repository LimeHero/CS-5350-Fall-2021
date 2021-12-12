# attempts to minimize the cost of w*values,
# where w is the desired vector we return
#
# r is the learning rate.
#
# descent_type must be either "sgd" or "grad",
# which stands for (regular) gradient descent
import random


def lms(values, labels, r, descent_type="sgd", print_info=False):
    if descent_type != "sgd" and descent_type != "grad":
        raise Exception("descent_type must be equal to \"sgd\" or \"grad\"")

    w = [0] * (len(values[0]) + 1)
    # value of w in the previous step to check for convergence
    w_ = [1] * len(w)

    num_gen = 0
    while True:
        num_gen += 1

        diff = 0
        for i in range(len(w)):
            diff += (w[i] - w_[i]) ** 2

        # if print_info:
        # print("convergence: " + str(diff))

        cost = 0
        for j in range(len(values)):
            dot_prod = 0
            for i in range(len(values[j])):
                dot_prod += w[i] * values[j][i]
            dot_prod += w[-1]
            cost += (labels[j] - dot_prod) ** 2

        if cost < 30.5:
            break

        if print_info and num_gen % 100 == 0:
            print(str(num_gen) + "," + str(cost))

        w_ = w.copy()

        gradient = [0] * len(w)
        if descent_type == "sgd":
            index = random.randrange(len(values))
            gradient = lms_neg_gradient(w, values[index], labels[index])

        if descent_type == "grad":
            for i in range(len(labels)):
                next_grad = lms_neg_gradient(w, values[i], labels[i])
                for j in range(len(next_grad)):
                    gradient[j] += next_grad[j]

        for i in range(len(w)):
            w[i] += gradient[i] * r

    return w


# helper function to find the *negative* gradient for a single
# element, which is how sgd is defined, or the total
# gradient is the sum
def lms_neg_gradient(w, value, label):
    grad = []

    # w^T \dot value
    dot_prod = 0
    for i in range(len(value)):
        dot_prod += w[i] * value[i]
    dot_prod += w[-1]

    for i in range(len(value)):
        grad.append((label - dot_prod) * value[i])

    grad.append(label - dot_prod)

    return grad
