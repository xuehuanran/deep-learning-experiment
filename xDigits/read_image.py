import tensorflow as tf

filename = 'mnist-2.png'
raw_img_data = tf.gfile.FastGFile(filename, mode='rb').read()

img_data = tf.image.decode_png(raw_img_data, channels=1)
img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
img_data = tf.image.resize_images(img_data, size=[28, 28])
