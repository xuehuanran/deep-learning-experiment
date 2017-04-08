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
for _ in range(4):
    w_conv1.append(weight_variable([5, 5, 1, 32]))
    b_conv1.append(bias_variable([32]))

for i in range(4):
    h_conv1.append(tf.nn.relu(conv2d(x_image, w_conv1[i]) + b_conv1[i]))
    h_pool1.append(max_pool_2x2(h_conv1[i]))


# the second convolution layer
w_conv2 = []
b_conv2 = []
h_conv2 = []
h_pool2 = []
for i in range(4):
    w_conv2.append(weight_variable([5, 5, 32, 64]))
    b_conv2.append(bias_variable([64]))
for i in range(4):
    h_conv2.append(tf.nn.relu(conv2d(h_pool1[i], w_conv2[i]) + b_conv2[i]))
    h_pool2.append(max_pool_2x2(h_conv2[i]))

# first fully-connected-layer
w_fc1 = []
b_fc1 = []
h_pool2_flat = []
h_fc1 = []
for i in range(4):
    w_fc1.append(weight_variable([7 * 7 * 64, 1024]))
    b_fc1.append(bias_variable([1024]))
for i in range(4):
    h_pool2_flat.append(tf.reshape(h_pool2[i], [-1, 7 * 7 * 64]))
    h_fc1.append(tf.nn.relu(tf.matmul(h_pool2_flat[i], w_fc1[i]) + b_fc1[i]))

# dropout
h_fc1_drop = []
keep_prob = tf.placeholder(tf.float32)
for i in range(4):
    h_fc1_drop.append(tf.nn.dropout(h_fc1[i], keep_prob))

# second fully-connected-layer(output)
w_fc2 = []
b_fc2 = []
y_conv = []
for i in range(4):
    w_fc2.append(weight_variable([1024, 10]))
    b_fc2.append(bias_variable([10]))
    y_conv.append(tf.matmul(h_fc1_drop[i], w_fc2[i]) + b_fc2[i])

# Train and Evaluate the Model
cross_entropy = []
for i in range(4):
    cross_entropy.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv[i])))

correct_prediction = []
accuracy = []
for i in range(4):
    correct_prediction.append(tf.equal(tf.argmax(y_conv[i], 1), tf.argmax(y_, 1)))
    accuracy.append(tf.reduce_mean(tf.cast(correct_prediction[i], tf.float32)))

session.run(tf.global_variables_initializer())

# record the loss of different algorithms
max_iteration = 50
loss0 = []
loss1 = []
loss2 = []
loss3 = []

output = open('training_data.txt', 'w')

for i in range(max_iteration):
    output.write('epoch : %i\n' % i)
    output.flush()

    parameter_list0 = [w_conv1[0], b_conv1[0], w_conv2[0], b_conv2[0], w_fc1[0], b_fc1[0], w_fc2[0], b_fc2[0]]
    parameter_list1 = [w_conv1[1], b_conv1[1], w_conv2[1], b_conv2[1], w_fc1[1], b_fc1[1], w_fc2[1], b_fc2[1]]
    parameter_list2 = [w_conv1[2], b_conv1[2], w_conv2[2], b_conv2[2], w_fc1[2], b_fc1[2], w_fc2[2], b_fc2[2]]
    parameter_list3 = [w_conv1[3], b_conv1[3], w_conv2[3], b_conv2[3], w_fc1[3], b_fc1[3], w_fc2[3], b_fc2[3]]

    batch = mnist.train.next_batch(50)
    loss0.append(cross_entropy[0].eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}))
    loss1.append(cross_entropy[1].eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}))
    loss2.append(cross_entropy[2].eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}))
    loss3.append(cross_entropy[3].eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}))

    train_step0 = training_algorithms.momentum(cross_entropy[0], i, parameter_list0)
    train_step1 = training_algorithms.momentum_modified(cross_entropy[1], i, parameter_list1)
    train_step2 = training_algorithms.nesterov(cross_entropy[2], i, parameter_list2)
    train_step3 = training_algorithms.nesterov_modified(cross_entropy[3], i, parameter_list3)

    session.run(train_step0, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    session.run(train_step1, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    session.run(train_step2, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    session.run(train_step3, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("momentum test accuracy %g" % (accuracy[0].eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})))
print("momentum_modified test accuracy %g" % (accuracy[1].eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})))
print("nesterov test accuracy %g" % (accuracy[2].eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})))
print("nesterov_modified test accuracy %g" % (accuracy[3].eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})))

np.save('loss0.npy', loss0)
np.save('loss1.npy', loss1)
np.save('loss2.npy', loss2)
np.save('loss3.npy', loss3)

end = time.time()
print('time : ', end - start, 's')


