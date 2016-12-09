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
	print logits, layer_biases, layer_weights
	return logits, layer_weights, layer_biases

def spherical_hinge_loss(feature_vec, labels=None, scope=None):
	"""Method that returns the loss tensor for spherical hinge loss.
	aArgs:
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
#	with tf.name_scope(scope, "spherical_hinge_loss", [feature_vec, labels]) as scope:
#	feature_vec.get_shape().assert_is_compatible_with(labels.get_shape())
	# We first need to convert binary labels to -1/1 labels (as floats).
	all_ones = tf.ones_like(labels)
#	labels_mod = tf.sub(tf.scalar_mul(2,tf.to_float(labels)), all_ones)
	labels_mod = tf.sub(tf.scalar_mul(2,tf.to_float(labels)), all_ones)
	norm_squared = tf.reduce_sum(tf.square(feature_vec), 1, keep_dims=True)
	print labels_mod, norm_squared, tf.nn.relu(tf.mul(tf.reshape(labels_mod,[-1, 1]),tf.sub(tf.reshape(all_ones,[-1, 1]), norm_squared)))
	return tf.nn.relu(tf.mul(tf.reshape(labels_mod,[-1, 1]),tf.sub(tf.reshape(all_ones,[-1, 1]), norm_squared)))

def spherical_softmax_loss(feature_vec, object_or_not):
	norm_squared = tf.reduce_sum(tf.square(feature_vec), 1, keep_dims=True)
	feature_size = 1
	layer_weights = weight_variable([feature_size, 1])
	layer_biases = bias_variable([1])
	logit1 = tf.matmul(norm_squared, layer_weights) + layer_biases
	logit2 = tf.matmul(norm_squared, -layer_weights) - layer_biases
	final_logit = tf.concat(1, [logit1, logit2])
	object_or_not_t = tf.reshape(object_or_not, [-1, 1])
	object_or_not_onehot = tf.concat(1, [object_or_not_t, tf.sub(tf.ones_like(object_or_not_t),object_or_not_t)])
	spherical_softmax_loss = tf.nn.softmax_cross_entropy_with_logits(final_logit, object_or_not_onehot)

	return spherical_softmax_loss, norm_squared



def knowledge_distillation_loss(soft_new, soft_old=None, T=2):
	all_T = tf.scalar_mul(1/float(T), tf.ones_like(soft_old))
	yP_new = tf.pow(soft_new,all_T)
	yP_old = tf.pow(soft_old,all_T)
	y_new = tf.mul(tf.reciprocal(tf.reduce_sum(yP_new)),yP_new)
	y_old = tf.mul(tf.reciprocal(tf.reduce_sum(yP_old)),yP_old)
	return tf.negative(tf.reduce_sum(tf.mul(y_old, tf.log(y_new))))

