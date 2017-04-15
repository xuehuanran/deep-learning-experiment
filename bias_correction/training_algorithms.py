import tensorflow as tf

t = tf.Variable(1, False)


def momentum(loss, mu):
    return tf.train.MomentumOptimizer(1e-3, mu).minimize(loss)


def momentum_modified(loss, mu):
    exp_mu = tf.train.exponential_decay(1.0, t, 1.0, mu)
    learning_rate = 1e-3 / (1.0 - exp_mu)
    opt = tf.train.MomentumOptimizer(learning_rate, mu)
    return opt.minimize(loss, t)
