import tensorflow as tf

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.01)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.01, shape = shape)
	return tf.Variable(initial)

def conv2d(x, W, stride):
	return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")


def add_last_layer(final_features, output_dim):
	feature_size = int(final_features.get_shape()[1])
	layer_weights = weight_variable([feature_size, output_dim])
	layer_biases = bias_variable([output_dim])
	logits = tf.matmul(final_features, layer_weights) + layer_biases
	return logits, layer_weights, layer_biases

'''
def add_last_layer(final_features, output_dim):
	with tf.name_scope('weights'):
		feature_size = int(final_features.get_shape()[1])
		layer_weights = tf.Variable(
		tf.truncated_normal([feature_size, output_dim], stddev=0.001), name='final_weights')
	with tf.name_scope('biases'):
		layer_biases = tf.Variable(tf.zeros([output_dim]), name='final_biases')
	with tf.name_scope('Wx_plus_b'):
		logits = tf.matmul(final_features, layer_weights,
					name='final_matmul') + layer_biases
		tf.histogram_summary('pre_activations', logits)
	return logits, layer_weights, layer_biases
'''