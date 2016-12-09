import tensorflow as tf
import numpy as np
import os, sys
from utils import *
from nn_utils import weight_variable, bias_variable, conv2d, max_pool_2x2, add_last_layer, spherical_hinge_loss, spherical_softmax_loss

class generic_model():
	def __init__(self, class_count):
		self.class_count = class_count

	def build_basic_graph(self, sess):
		W_conv1 = weight_variable([8, 8, 3, 32])
		b_conv1 = bias_variable([32])

		W_conv2 = weight_variable([4, 4, 32, 64])
		b_conv2 = bias_variable([64])

		W_conv3 = weight_variable([3, 3, 64, 64])
		b_conv3 = bias_variable([64])

		W_fc1 = weight_variable([6400, 512])
		b_fc1 = bias_variable([512])

		W_fc2 = weight_variable([512, self.class_count])
		b_fc2 = bias_variable([self.class_count])

		imgTensor = tf.placeholder("float", [None, 80, 80, 3])
		labelTensor = tf.placeholder("float", [None, self.class_count])
		object_or_not = tf.placeholder("float", [None])

	    # hidden layers
		conv1 = tf.nn.relu(conv2d(imgTensor, W_conv1, 4) + b_conv1)

		conv2 = tf.nn.relu(conv2d(conv1, W_conv2, 2) + b_conv2)

		conv3 = tf.nn.relu(conv2d(conv2, W_conv3, 1) + b_conv3)

		conv3_flat = tf.reshape(conv3, [-1, 6400])

		h_fc1 = tf.nn.relu(tf.matmul(conv3_flat, W_fc1) + b_fc1)

		scores = tf.matmul(h_fc1, W_fc2) + b_fc2

		return object_or_not, labelTensor, imgTensor, scores, h_fc1

	def build_graph_for_target(self, sess, labelTensor, scores, h_fc1, object_or_not):
		cross_entropy = tf.reduce_mean(tf.mul(tf.nn.softmax_cross_entropy_with_logits(scores, labelTensor), object_or_not))

		# sphere_loss_beforeMean = spherical_hinge_loss(h_fc1, object_or_not)
		sphere_loss_beforeMean, norm_squared = spherical_softmax_loss(h_fc1, object_or_not)
		sphere_loss = tf.reduce_mean(sphere_loss_beforeMean)

		lambda_train_s = 1
		total_loss = tf.add(cross_entropy,tf.scalar_mul(lambda_train_s, sphere_loss))

		train_step = tf.train.AdamOptimizer(1e-5).minimize(total_loss)
		sess.run(tf.initialize_all_variables())
		return cross_entropy, sphere_loss, train_step, norm_squared


	
