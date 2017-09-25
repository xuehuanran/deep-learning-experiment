from tensorflow.python.training.gradient_descent import GradientDescentOptimizer
import tensorflow as tf

last_momentum1 = []
last_momentum2 = []
last_momentum3 = []
last_momentum4 = []

last_n1 = []
last_n2 = []
last_n3 = []
last_n4 = []

last_g = []
learning_rate1 = 1e-3
learning_rate2 = 1e-3


def RMSProp(loss,parameter_list):

    return tf.train.RMSPropOptimizer(1e-3).minimize(loss)
    # opt = GradientDescentOptimizer(1e-3)
    # grads_and_vars = opt.compute_gradients(loss, parameter_list)
    # capped_grads_and_vars = []
    # middle = []
    #
    # for i in range(len(grads_and_vars)):
    #     gradient = grads_and_vars[i][0]
    #     variable = grads_and_vars[i][1]
    #     if len(last_n1) != 0:
    #         n = 0.8 * tf.multiply(gradient, gradient) + 0.2 * last_n1[i-1]
    #         middle.append(n)
    #         momentum = gradient/(tf.sqrt(n) + 0.001)
    #         capped_grads_and_vars.append((momentum, variable))
    #     else:
    #         n = tf.multiply(gradient, gradient)
    #         middle.append(n)
    #         momentum = gradient / (tf.sqrt(n) + 0.001)
    #         capped_grads_and_vars.append((momentum, variable))
    #
    # if len(last_n1) != 0:
    #     for i in range(len(capped_grads_and_vars)):
    #         last_n1[i] = middle[i]
    # else:
    #     for i in range(len(capped_grads_and_vars)):
    #         last_n1.append(middle[i])
    #
    #
    # return opt.apply_gradients(capped_grads_and_vars)


def RMSProp_Mom(loss, parameter_list):

    mu = 0.9  # the parameter of the momentum, always be 0.9
    opt = GradientDescentOptimizer(1e-3)
    grads_and_vars = opt.compute_gradients(loss, parameter_list)
    capped_grads_and_vars = []
    middle = []

    for i in range(len(grads_and_vars)):
        gradient = grads_and_vars[i][0]
        variable = grads_and_vars[i][1]
        if len(last_n2) != 0:
            n = 0.8 * tf.multiply(gradient, gradient) + 0.2 * last_n2[i-1]
            middle.append(n)
            momentum = 0.9 * last_momentum2[i-1] + gradient/(tf.sqrt(n) + 0.001)
            capped_grads_and_vars.append((momentum, variable))
        else:
            n = tf.multiply(gradient, gradient)
            middle.append(n)
            momentum = gradient / (tf.sqrt(n) + 0.001)
            capped_grads_and_vars.append((momentum, variable))

    if len(last_momentum2) != 0:
        for i in range(len(capped_grads_and_vars)):
            last_momentum2[i] = capped_grads_and_vars[i][0]
    else:
        for i in range(len(capped_grads_and_vars)):
            last_momentum2.append(capped_grads_and_vars[i][0])

    if len(last_n2) != 0:
        for i in range(len(capped_grads_and_vars)):
            last_n2[i] = middle[i]
    else:
        for i in range(len(capped_grads_and_vars)):
            last_n2.append(middle[i])

    return opt.apply_gradients(capped_grads_and_vars)

def RMSProp_BB1(loss,parameter_list,learning_rate1):

    opt = GradientDescentOptimizer(learning_rate1)
    grads_and_vars = opt.compute_gradients(loss, parameter_list)
    capped_grads_and_vars = []
    middle = []

    for i in range(len(grads_and_vars)):
        gradient = grads_and_vars[i][0]
        variable = grads_and_vars[i][1]
        if len(last_n3) != 0:
            n = 0.8 * tf.multiply(gradient, gradient) + 0.2 * last_n3[i-1]
            middle.append(n)
            momentum = gradient/(tf.sqrt(n) + 0.001)
            capped_grads_and_vars.append((momentum, variable))
            s = variable-grads_and_vars[i-1][1]
            y = gradient = grads_and_vars[i-1][0]
            learning_rate1 = (s*y) / (y*y)
            print(learning_rate1)
        else:
            n = tf.multiply(gradient, gradient)
            middle.append(n)
            momentum = gradient / (tf.sqrt(n) + 0.001)
            capped_grads_and_vars.append((momentum, variable))
            s = variable
            y = gradient
            learning_rate1 = (s*y)/(y*y)


    if len(last_n3) != 0:
        for i in range(len(capped_grads_and_vars)):
            last_n3[i] = middle[i]
    else:
        for i in range(len(capped_grads_and_vars)):
            last_n3.append(middle[i])

    return opt.apply_gradients(capped_grads_and_vars)


def RMSProp_BB2(loss,parameter_list,learning_rate2):

    opt = GradientDescentOptimizer(learning_rate2)
    grads_and_vars = opt.compute_gradients(loss, parameter_list)
    capped_grads_and_vars = []
    middle = []

    for i in range(len(grads_and_vars)):
        gradient = grads_and_vars[i][0]
        variable = grads_and_vars[i][1]
        if len(last_n4) != 0:
            n = 0.8 * tf.multiply(gradient, gradient) + 0.2 * last_n4[i-1]
            middle.append(n)
            momentum = gradient/(tf.sqrt(n) + 0.001)
            capped_grads_and_vars.append((momentum, variable))
        else:
            n = tf.multiply(gradient, gradient)
            middle.append(n)
            momentum = gradient / (tf.sqrt(n) + 0.001)
            capped_grads_and_vars.append((momentum, variable))

    if len(last_n4) != 0:
        for i in range(len(capped_grads_and_vars)):
            last_n4[i] = middle[i]
    else:
        for i in range(len(capped_grads_and_vars)):
            last_n4.append(middle[i])

    return opt.apply_gradients(capped_grads_and_vars)
