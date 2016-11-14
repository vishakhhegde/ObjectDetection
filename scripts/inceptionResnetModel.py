import tensorflow as tf
import numpy as np
import os, sys
from tensorflow.python.platform import gfile

# model_order is the name of the model used
# network_id
# image_dir
# ckpt_path
# input_graph_path
# output_ckpt_dir
# output_graph_path
# session
# class_count

class generic_model():
	def __init__(self, model_order, network_id, image_dir, input_ckpt_path, output_ckpt_dir, input_graph_path, output_graph_path, sess, class_count):
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

	