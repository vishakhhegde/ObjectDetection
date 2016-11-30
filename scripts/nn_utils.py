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

def spherical_hinge_loss(feature_vec, labels=None, scope=None):
  """Method that returns the loss tensor for spherical hinge loss.
  Args:
    feature_vec: the features, a float tensor.
    labels: The ground truth output tensor. Its shape should match the shape of
      logits. The values of the tensor are expected to be 0.0 or 1.0.
    scope: The scope for the operations performed in computing the loss.
  Returns:
    A `Tensor` of same shape as feature_vec and target representing the loss values
      across the batch.
  Raises:
    ValueError: If the shapes of `feature_vec` and `labels` don't match.
  """
  with tf.name_scope(scope, "spherical_hinge_loss", [feature_vec, labels]) as scope:
    feature_vec.get_shape().assert_is_compatible_with(labels.get_shape())
    # We first need to convert binary labels to -1/1 labels (as floats).
    labels = tf.to_float(labels)
    all_ones = tf.ones_like(labels)
    labels = tf.sub(2 * tf.sign(labels), all_ones)
	norm_squared = tf.sum(tf.square(feature_vec))
    return tf.nn.relu(tf.mul(labels,tf.sub(all_ones, norm_squared)))
