from tensorflow.python.training.gradient_descent import GradientDescentOptimizer
import tensorflow as tf

last_momentum0 = []
last_momentum1 = []

last_n0 = []
last_n1 = []


def Ada(loss, parameter_list):
    return tf.train.AdagradOptimizer(1e-3).minimize(loss)
    # opt = GradientDescentOptimizer(1e-3)
    # grads_and_vars = opt.compute_gradients(loss, parameter_list)
    # capped_grads_and_vars = []
    # middle = []
    #
    # for i in range(len(grads_and_vars)):
    #     gradient = grads_and_vars[i][0]
    #     variable = grads_and_vars[i][1]
    #     if len(last_n0)!=0:
    #         n = tf.multiply(gradient, gradient) + last_n0[i-1]
    #         middle.append(n)
    #         momentum = gradient/(tf.sqrt(n) + 0.001)
    #         capped_grads_and_vars.append((momentum, variable))
    #
    #     else:
    #         n = tf.multiply(gradient, gradient)
    #         middle.append(n)
    #         momentum = gradient / (tf.sqrt(n) + 0.001)
    #         capped_grads_and_vars.append((momentum, variable))
    #
    # if len(last_n0) != 0:
    #     for i in range(len(capped_grads_and_vars)):
    #         last_n0[i] = middle[i]
    # else:
    #     for i in range(len(capped_grads_and_vars)):
    #         last_n0.append(middle[i])
    #
    #
    # return opt.apply_gradients(capped_grads_and_vars)


def Ada_Mom(loss, parameter_list):

    mu = 0.9  # the parameter of the momentum, always be 0.9
    opt = GradientDescentOptimizer(1e-3)
    grads_and_vars = opt.compute_gradients(loss, parameter_list)
    capped_grads_and_vars = []
    middle = []

    for i in range(len(grads_and_vars)):
        gradient = grads_and_vars[i][0]
        variable = grads_and_vars[i][1]
        if len(last_n1) != 0:
            n = tf.multiply(gradient, gradient) + last_n1[i-1]
            middle.append(n)
            momentum = 0.9 * last_momentum1[i-1] + gradient/(tf.sqrt(n) + 0.001)
            capped_grads_and_vars.append((momentum, variable))
        else:
            n = tf.multiply(gradient, gradient)
            middle.append(n)
            momentum = gradient / (tf.sqrt(n) + 0.001)
            capped_grads_and_vars.append((momentum, variable))

    if len(last_momentum1) != 0:
        for i in range(len(capped_grads_and_vars)):
            last_momentum1[i] = capped_grads_and_vars[i][0]
    else:
        for i in range(len(grads_and_vars)):
            last_momentum1.append(capped_grads_and_vars[i][0])

    if len(last_n1) != 0:
        for i in range(len(capped_grads_and_vars)):
            last_n1[i] = middle[i]
    else:
        for i in range(len(grads_and_vars)):
            last_n1.append(middle[i])

    return opt.apply_gradients(capped_grads_and_vars)
