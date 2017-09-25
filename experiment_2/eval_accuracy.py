import tensorflow as tf
import cifar10_input


def conv2d(input_x, input_w):
    return tf.nn.conv2d(input_x, input_w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(input_x):
    return tf.nn.max_pool(input_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

w_conv1 = tf.Variable(tf.zeros([5, 5, 3, 64]), name='w_conv1')
b_conv1 = tf.Variable(tf.zeros([64]), name='b_conv1')
w_conv2 = tf.Variable(tf.zeros([5, 5, 64, 64]), name='w_conv2')
b_conv2 = tf.Variable(tf.zeros([64]), name='b_conv2')

w_fc1 = tf.Variable(tf.zeros([8 * 8 * 64, 1024]), name='w_fc1')
b_fc1 = tf.Variable(tf.zeros([1024]), name='b_fc1')
w_fc2 = tf.Variable(tf.zeros([1024, 10]), name='w_fc2')
b_fc2 = tf.Variable(tf.zeros([10]), name='b_fc2')


session = tf.InteractiveSession()

saver = tf.train.Saver()
saver.restore(session, 'output/rmsprop/rmsProp_params.ckpt')

y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.placeholder(tf.float32, [None, 32, 32, 3])

h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

test_images, test_labels = cifar10_input.load_cifar10(is_train=False)

print("test accuracy %g\n" % (
    accuracy.eval(feed_dict={x_image: test_images, y_: test_labels, keep_prob: 1.0})))

