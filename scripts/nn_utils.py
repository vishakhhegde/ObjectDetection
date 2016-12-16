import tensorflow as tf
import numpy as np

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
	all_ones = tf.ones_like(labels)
	labels_mod = tf.sub(tf.scalar_mul(2,tf.to_float(labels)), all_ones)
	norm_squared = tf.reduce_sum(tf.square(feature_vec), 1, keep_dims=True)
	spherical_softmax_loss = tf.nn.relu(tf.mul(tf.reshape(labels_mod,[-1, 1]),tf.sub(tf.reshape(all_ones,[-1, 1]), norm_squared)))
	return spherical_softmax_loss, norm_squared

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
	return spherical_softmax_loss, norm_squared, final_logit

def knowledge_distillation_loss(scores_new, scores_old=None, T=1):
	all_T = tf.scalar_mul(1/float(T), tf.ones_like(scores_old))
	yP_new = tf.pow(tf.nn.softmax(scores_new),all_T)
	yP_old = tf.pow(tf.nn.softmax(scores_old),all_T)
	y_new = tf.mul(tf.div(1.0, tf.add(0.0001, tf.reduce_sum(yP_new, 1, keep_dims=True))),yP_new)
	y_old = tf.mul(tf.div(1.0, tf.add(0.0001, tf.reduce_sum(yP_old, 1, keep_dims=True))),yP_old)
	return tf.scalar_mul(-1.0, tf.reduce_sum(tf.mul(y_old, tf.log(y_new))))

def test_object_detection_spherical_softmax(scores_class, score_object, object_or_not, label_inputs_one_hot):
	test_obj_or_not_neg = np.argmax(score_object, 1)
	print(score_object)
	test_obj_or_not = np.subtract(np.ones_like(test_obj_or_not_neg), test_obj_or_not_neg)
	print(test_obj_or_not)
	object_labels = np.argmax(label_inputs_one_hot, 1)
	test_object_labels = np.argmax(scores_class[:][0:20], 1)
	print(test_object_labels) 
	object_detect_vector = [int(x) for x in test_obj_or_not == object_or_not]
	print(object_detect_vector)
	object_detection_score = np.sum(object_detect_vector) 
	
	object_classification_vector = np.multiply([int(x) for x in test_object_labels == object_labels],\
		np.multiply(object_detect_vector, object_or_not))
	object_classification_score = np.sum(object_classification_vector)
	return object_detection_score, object_classification_score

def test_object_detection_spherical_hinge(scores_class, norm_squared, object_or_not, label_inputs_one_hot):
	test_obj_or_not = [int(x) for x in norm_squared >= 1]
	print(norm_squared)
	# test_obj_or_not = np.subtract(np.ones_like(test_obj_or_not_neg), test_obj_or_not_neg)
	object_labels = np.argmax(label_inputs_one_hot, 1)
	test_object_labels = np.argmax(scores_class[:][0:20], 1) 
	object_detect_vector = [int(x) for x in np.equal(test_obj_or_not, object_or_not)]
	object_detection_score = np.sum(object_detect_vector)
	# print(object_detect_vector) 
	# print(test_obj_or_not)
	# print(object_or_not)
	print(scores_class)
	print(object_labels)
	print(test_object_labels)
	object_classification_vector = np.multiply([int(x) for x in np.equal(test_object_labels,object_labels)],\
		np.multiply(object_detect_vector, object_or_not))
	print(object_classification_vector)
	object_classification_score = np.sum(object_classification_vector)
	return object_detection_score, object_classification_score

def test_object_detection_softmax(scores_class, object_or_not, label_inputs_one_hot):
	object_labels = np.argmax(label_inputs_one_hot, 1)
	test_object_labels = np.argmax(scores_class, 1) 
	object_detection_score = np.sum((object_labels < 20) == object_or_not)
	
	object_net_vector = test_object_labels == object_labels
	object_classification_vector = np.multiply(object_net_vector, object_or_not)
	object_classification_score = np.sum(object_classification_vector)
	return object_detection_score, object_classification_score