import numpy as np
import tensorflow as tf
import read_image


def conv2d(input_x, input_w):
    return tf.nn.conv2d(input_x, input_w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(input_x):
    return tf.nn.max_pool(input_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

w_conv1_np = np.load('parameters/w_conv1.npy')
b_conv1_np = np.load('parameters/b_conv1.npy')
w_conv2_np = np.load('parameters/w_conv2.npy')
b_conv2_np = np.load('parameters/b_conv2.npy')
w_fc1_np = np.load('parameters/w_fc1.npy')
b_fc1_np = np.load('parameters/b_fc1.npy')
w_fc2_np = np.load('parameters/w_fc2.npy')
b_fc2_np = np.load('parameters/b_fc2.npy')

session = tf.InteractiveSession()

x_init = np.zeros([1, 28, 28, 1])
x_init[0] = read_image.img_data.eval()
x = tf.constant(x_init, dtype=tf.float32)

# the first convolution layer
w_conv1 = tf.Variable(w_conv1_np[1])
b_conv1 = tf.Variable(b_conv1_np[1])

h_conv1 = tf.nn.relu(conv2d(x, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# the second convolution layer
w_conv2 = tf.Variable(w_conv2_np[1])
b_conv2 = tf.Variable(b_conv2_np[1])

h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# first fully-connected-layer
w_fc1 = tf.Variable(w_fc1_np[1])
b_fc1 = tf.Variable(b_fc1_np[1])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# second fully-connected-layer(output)
w_fc2 = tf.Variable(w_fc2_np[1])
b_fc2 = tf.Variable(b_fc2_np[1])
y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2
prob_tf = tf.nn.softmax(y_conv)

session.run(tf.global_variables_initializer())

prob = prob_tf.eval(feed_dict={keep_prob: 1.0})

for i in range(10):
   print('%d : %f' % (i, prob[0][i]))
