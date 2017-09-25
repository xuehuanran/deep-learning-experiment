import tensorflow as tf
import cifar10_input
import training_algorithms
import numpy as np


def weight_variable(shape, stdeev, name):
    initial = tf.truncated_normal(shape, stddev=stdeev)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(input_x, input_w):
    return tf.nn.conv2d(input_x, input_w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(input_x):
    return tf.nn.max_pool(input_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


session = tf.InteractiveSession()

x_image = tf.placeholder(tf.float32, [None, 32, 32, 3])
y_ = tf.placeholder(tf.float32, [None, 10])

# the first convolution layer
w_conv1 = weight_variable([5, 5, 3, 64], stdeev=5e-2, name='w_conv1')
b_conv1 = bias_variable([64], name='b_conv1')

h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# the second convolution layer
w_conv2 = weight_variable([5, 5, 64, 64], stdeev=5e-2, name='w_conv2')
b_conv2 = bias_variable([64], name='b_conv2')

h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# first fully-connected-layer
w_fc1 = weight_variable([8 * 8 * 64, 1024], stdeev=0.04, name='w_fc1')
b_fc1 = bias_variable([1024], name='b_fc1')

h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# second fully-connected-layer(output)
w_fc2 = weight_variable([1024, 10], stdeev=0.04, name='w_fc2')
b_fc2 = bias_variable([10], name='b_fc2')
y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

# Train and Evaluate the Model
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = training_algorithms.momentum(cross_entropy, 0.9)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session.run(tf.global_variables_initializer())
train_images = []
train_labels = []
index = 0

batch_accuracy = []
loss = []
for i in range(20):
    if i % 1000 == 0:
        index = 0
        train_images, train_labels = cifar10_input.load_cifar10(is_train=True)

    train_accuracy = accuracy.eval(feed_dict={x_image: train_images[index: (index + 50)],
                                              y_: train_labels[index: (index + 50)],
                                              keep_prob: 1.0})
    print("step %d, training accuracy %g" % (i, train_accuracy))

    batch_accuracy.append(train_accuracy)
    loss.append(cross_entropy.eval(feed_dict={x_image: train_images[index: (index + 50)],
                                              y_: train_labels[index: (index + 50)],
                                              keep_prob: 1.0}))

    session.run(train_step, feed_dict={x_image: train_images[index: index + 50],
                                       y_: train_labels[index: index + 50],
                                       keep_prob: 0.5})
    index = index + 50

saver = tf.train.Saver()
saver.save(session, 'output/momentum_params.ckpt')

print(batch_accuracy.__class__)

np.save('output/batch_accuracy.npy', batch_accuracy)
np.save('output/loss.npy', loss)

test_images, test_labels = cifar10_input.load_cifar10(is_train=False)
print("test accuracy %g" % (accuracy.eval(feed_dict={x_image: test_images, y_: test_labels, keep_prob: 1.0})))
