import tensorflow as tf
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
import training_algorithms
import time
import numpy as np


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(input_x, input_w):
    return tf.nn.conv2d(input_x, input_w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(input_x):
    return tf.nn.max_pool(input_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

start = time.time()

algorithm_num = 2

mnist = read_data_sets('../dataset/mnist', one_hot=True)

session = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

x_image = tf.reshape(x, [-1, 28, 28, 1])

# the first convolution layer
w_conv1 = []
b_conv1 = []
h_conv1 = []
h_pool1 = []
for _ in range(algorithm_num):
    w_conv1.append(weight_variable([5, 5, 1, 32]))
    b_conv1.append(bias_variable([32]))

for i in range(algorithm_num):
    h_conv1.append(tf.nn.relu(conv2d(x_image, w_conv1[i]) + b_conv1[i]))
    h_pool1.append(max_pool_2x2(h_conv1[i]))


# the second convolution layer
w_conv2 = []
b_conv2 = []
h_conv2 = []
h_pool2 = []
for i in range(algorithm_num):
    w_conv2.append(weight_variable([5, 5, 32, 64]))
    b_conv2.append(bias_variable([64]))
for i in range(algorithm_num):
    h_conv2.append(tf.nn.relu(conv2d(h_pool1[i], w_conv2[i]) + b_conv2[i]))
    h_pool2.append(max_pool_2x2(h_conv2[i]))

# first fully-connected-layer
w_fc1 = []
b_fc1 = []
h_pool2_flat = []
h_fc1 = []
for i in range(algorithm_num):
    w_fc1.append(weight_variable([7 * 7 * 64, 1024]))
    b_fc1.append(bias_variable([1024]))
for i in range(algorithm_num):
    h_pool2_flat.append(tf.reshape(h_pool2[i], [-1, 7 * 7 * 64]))
    h_fc1.append(tf.nn.relu(tf.matmul(h_pool2_flat[i], w_fc1[i]) + b_fc1[i]))

# dropout
h_fc1_drop = []
keep_prob = tf.placeholder(tf.float32)
for i in range(algorithm_num):
    h_fc1_drop.append(tf.nn.dropout(h_fc1[i], keep_prob))

# second fully-connected-layer(output)
w_fc2 = []
b_fc2 = []
y_conv = []
for i in range(algorithm_num):
    w_fc2.append(weight_variable([1024, 10]))
    b_fc2.append(bias_variable([10]))
    y_conv.append(tf.matmul(h_fc1_drop[i], w_fc2[i]) + b_fc2[i])

# Train and Evaluate the Model
cross_entropy = []
for i in range(algorithm_num):
    cross_entropy.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv[i])))

correct_prediction = []
accuracy = []
for i in range(algorithm_num):
    correct_prediction.append(tf.equal(tf.argmax(y_conv[i], 1), tf.argmax(y_, 1)))
    accuracy.append(tf.reduce_mean(tf.cast(correct_prediction[i], tf.float32)))

train_step0 = training_algorithms.momentum(cross_entropy[0], 0.9)
train_step1 = training_algorithms.momentum_modified(cross_entropy[1], 0.9)


session.run(tf.global_variables_initializer())

# record the loss of different algorithms
max_iteration = 1100 * 50
loss0 = []
loss1 = []

for i in range(max_iteration):
    print('epoch : %i' % i)
    batch = mnist.train.next_batch(50)

    if (i + 1) % 1100 == 0:
        loss0.append(cross_entropy[0].eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}))
        loss1.append(cross_entropy[1].eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}))

    session.run(train_step0, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    session.run(train_step1, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

np.save('output/loss0.npy', loss0)
np.save('output/loss1.npy', loss1)

np.save('output/w_conv1.npy', session.run(w_conv1))
np.save('output/b_conv1.npy', session.run(b_conv1))
np.save('output/w_conv2.npy', session.run(w_conv2))
np.save('output/b_conv2.npy', session.run(b_conv2))
np.save('output/w_fc1.npy', session.run(w_fc1))
np.save('output/b_fc1.npy', session.run(b_fc1))
np.save('output/w_fc2.npy', session.run(w_fc2))
np.save('output/b_fc2.npy', session.run(b_fc2))


end = time.time()
print('time : %gs' % (end - start))
