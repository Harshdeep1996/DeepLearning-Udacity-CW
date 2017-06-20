# So for making a convolutional neural network in tensorflow, there are 2 main
# functions.

# tf.nn.conv2d()
# tf.nn.bias_add()

import tensorflow as tf

image_height = 10
image_width = 10
image_channels = 3

filter_height = 5
filter_width = 5
filter_channels = 64

inputs = tf.placeholder(
    tf.place32, shape=[None, image_height, image_width, image_channels]
)
# Take the weights for each filter channels - 64
weights = tf.Variable(tf.truncated_normal(
    [filter_height, filter_width, image_channels, filter_channels])
)
bias = tf.Variable(tf.zeros(filter_channels))

# Stride for moving along X,Y axes
# Same padding for adding rows along the image
conv_layer = tf.nn.conv2d(inputs, weights, strides=[1, 2, 2, 1], padding='SAME')
conv_layer = tf.bias_add(bias)
conv_layer = tf.nn.relu(conv_layer)

# Pools the values together by taking the max value in the neigboring cells
conv_layer = tf.nn.max_pool(
    conv_layer, k_size=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'
)
