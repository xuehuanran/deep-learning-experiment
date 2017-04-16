import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets


def conv2d(input_x, input_w):
    return tf.nn.conv2d(input_x, input_w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(input_x):
    return tf.nn.max_pool(input_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

w_conv1_np = np.load('output/w_conv1.npy')
b_conv1_np = np.load('output/b_conv1.npy')
w_conv2_np = np.load('output/w_conv2.npy')
b_conv2_np = np.load('output/b_conv2.npy')
w_fc1_np = np.load('output/w_fc1.npy')
b_fc1_np = np.load('output/b_fc1.npy')
w_fc2_np = np.load('output/w_fc2.npy')
b_fc2_np = np.load('output/b_fc2.npy')

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
for i in range(algorithm_num):
    w_conv1.append(tf.Variable(w_conv1_np[i]))
    b_conv1.append(tf.Variable(b_conv1_np[i]))

for i in range(algorithm_num):
    h_conv1.append(tf.nn.relu(conv2d(x_image, w_conv1[i]) + b_conv1[i]))
    h_pool1.append(max_pool_2x2(h_conv1[i]))


# the second convolution layer
w_conv2 = []
b_conv2 = []
h_conv2 = []
h_pool2 = []
for i in range(algorithm_num):
    w_conv2.append(tf.Variable(w_conv2_np[i]))
    b_conv2.append(tf.Variable(b_conv2_np[i]))
for i in range(algorithm_num):
    h_conv2.append(tf.nn.relu(conv2d(h_pool1[i], w_conv2[i]) + b_conv2[i]))
    h_pool2.append(max_pool_2x2(h_conv2[i]))

# first fully-connected-layer
w_fc1 = []
b_fc1 = []
h_pool2_flat = []
h_fc1 = []
for i in range(algorithm_num):
    w_fc1.append(tf.Variable(w_fc1_np[i]))
    b_fc1.append(tf.Variable(b_fc1_np[i]))
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
    w_fc2.append(tf.Variable(w_fc2_np[i]))
    b_fc2.append(tf.Variable(b_fc2_np[i]))
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

session.run(tf.global_variables_initializer())

output = open('output/test_accuracy.txt', 'w')

output.write("momentum test accuracy %g\n" % (
    accuracy[0].eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})))
output.write("momentum_modified test accuracy %g\n" % (
    accuracy[1].eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})))


output.flush()
