import tensorflow as tf
import numpy as np
import os, sys
from genericModel import *
from utils import *
from nn_utils import weight_variable, bias_variable, conv2d, max_pool_2x2, add_last_layer
# model_order is the name of the model used
# network_id
# image_dir
# ckpt_path
# input_graph_path
# output_ckpt_dir
# output_graph_path
# session
# class_count

TENSOR_TO_BE_GOT = 'InceptionResnetV2/Logits/Flatten/Reshape'
GROUND_TRUTH_TENSOR_NAME = 'ground_truth'
FINAL_TENSOR_NAME = 'final_tensor'
resized_input_tensor_name = 'ResizeBilinear'

class generic_model():
	def __init__(self, model_order, network_id, image_dir, input_ckpt_path, output_ckpt_dir, input_graph_path, output_graph_path, class_count):
		self.model_order = model_order
		self.network_id = network_id
		self.image_dir = image_dir
		self.input_ckpt_path = input_ckpt_path
		self.output_ckpt_dir = output_ckpt_dir
		self.input_graph_path = input_graph_path
		self.output_graph_path = output_graph_path
		self.class_count = class_count

	def build_basic_graph(self, sess):
		saver = tf.train.import_meta_graph(self.input_graph_path)
		saver.restore(sess, self.input_ckpt_path)

	def build_graph_for_target(self, sess):
		# This is how the ground truth is fed
		ground_truth_tensor = tf.placeholder(tf.float32, shape=(None, self.class_count),name=GROUND_TRUTH_TENSOR_NAME)
		
		# Get the relevant tensor from the graph
		inception_feature_tensor = sess.graph.get_tensor_by_name(ensure_name_has_port(TENSOR_TO_BE_GOT))

		# Add final layer for classification
		logits, _, _ = add_last_layer(inception_feature_tensor, self.class_count)
		final_tensor = tf.nn.softmax(logits, name=FINAL_TENSOR_NAME)
		lambda_train_s = 1.0  
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(final_tensor, ground_truth_tensor))
		sphere_loss = tf.reduce_mean(spherical_hinge_loss(inception_feature_tensor, ground_truth_tensor))
		train_step = tf.train.AdamOptimizer(1e-1).minimize(tf.sum(cross_entropy,tf.mul(lambda_train_s, sphere_loss)))
		sess.run(tf.initialize_all_variables())
		return (cross_entropy, sphere_loss)

	