from tensorflow.python.training.gradient_descent import GradientDescentOptimizer
import tensorflow as tf

last_momentum0 = []
last_momentum1 = []
last_momentum2 = []
last_momentum3 = []

mu = 0.5  # the parameter of the momentum, always be 0.9


def momentum(loss, t, parameter_list):
    opt = GradientDescentOptimizer(1e-3)
    grads_and_vars = opt.compute_gradients(loss, parameter_list)
    capped_grads_and_vars = []
    for i in range(len(grads_and_vars)):
        gradient = grads_and_vars[i][0]
        variable = grads_and_vars[i][1]
        if t != 0:
            momentum = gradient + mu * last_momentum0[i]
            capped_grads_and_vars.append((momentum, variable))
        else:
            momentum = gradient
            capped_grads_and_vars.append((momentum, variable))
    if t != 0:
        for i in range(len(grads_and_vars)):
            last_momentum0[i] = capped_grads_and_vars[i][0]
    else:
        for i in range(len(grads_and_vars)):
            last_momentum0.append(capped_grads_and_vars[i][0])
    return opt.apply_gradients(capped_grads_and_vars)


def momentum_modified(loss, t, parameter_list):
    opt = GradientDescentOptimizer(1e-3)
    grads_and_vars = opt.compute_gradients(loss, parameter_list)
    capped_grads_and_vars = []
    for i in range(len(grads_and_vars)):
        gradient = grads_and_vars[i][0]
        variable = grads_and_vars[i][1]
        if t != 0:
            momentum = (gradient + mu * last_momentum1[i]) / (1 - mu ** t)
            capped_grads_and_vars.append((momentum, variable))
        else:
            momentum = gradient
            capped_grads_and_vars.append((momentum, variable))
    if t != 0:
        for i in range(len(grads_and_vars)):
            last_momentum1[i] = capped_grads_and_vars[i][0]
    else:
        for i in range(len(grads_and_vars)):
            last_momentum1.append(capped_grads_and_vars[i][0])
    return opt.apply_gradients(capped_grads_and_vars)


def nesterov(loss, t, parameter_list):
    opt = GradientDescentOptimizer(1e-3)
    if t != 0:
        for i in range(len(parameter_list)):
            variable = parameter_list[i]
            variable -= 1e-3 * mu * last_momentum2[i]
            parameter_list[i].assign(variable)
    grads_and_vars = opt.compute_gradients(loss, parameter_list)
    capped_grads_and_vars = []
    for i in range(len(grads_and_vars)):
        gradient = grads_and_vars[i][0]
        variable = grads_and_vars[i][1]
        if t != 0:
            momentum = gradient + mu * last_momentum2[i]
            capped_grads_and_vars.append((momentum, variable))
        else:
            momentum = gradient
            capped_grads_and_vars.append((momentum, variable))
    if t != 0:
        for i in range(len(grads_and_vars)):
            last_momentum2[i] = capped_grads_and_vars[i][0]
    else:
        for i in range(len(grads_and_vars)):
            last_momentum2.append(capped_grads_and_vars[i][0])
    return opt.apply_gradients(capped_grads_and_vars)


def nesterov_modified(loss, t, parameter_list=None):
    opt = GradientDescentOptimizer(1e-3)
    if t != 0:
        for i in range(len(parameter_list)):
            variable = parameter_list[i]
            variable -= 1e-3 * mu * last_momentum3[i]
            parameter_list[i].assign(variable)
    grads_and_vars = opt.compute_gradients(loss, parameter_list)
    capped_grads_and_vars = []
    for i in range(len(grads_and_vars)):
        gradient = grads_and_vars[i][0]
        variable = grads_and_vars[i][1]
        if t != 0:
            momentum = (gradient + mu * last_momentum3[i]) / (1 - mu ** t)
            capped_grads_and_vars.append((momentum, variable))
        else:
            momentum = gradient
            capped_grads_and_vars.append((momentum, variable))
    if t != 0:
        for i in range(len(grads_and_vars)):
            last_momentum3[i] = capped_grads_and_vars[i][0]
    else:
        for i in range(len(grads_and_vars)):
            last_momentum3.append(capped_grads_and_vars[i][0])
    return opt.apply_gradients(capped_grads_and_vars)


