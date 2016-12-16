import tensorflow as tf
import numpy as np
from numpy import *
import os, sys
from utils import *
from nn_utils import weight_variable, bias_variable, conv2d, max_pool_2x2, add_last_layer, spherical_hinge_loss, spherical_softmax_loss
from nn_utils import knowledge_distillation_loss

# BOTTLENECK_TENSOR_NAME = 'InceptionResnetV2/Logits/Flatten/Reshape'
# RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear'
# GROUND_TRUTH_TENSOR_NAME = 'ground_truth'
# INCEPTIONRESNETV2_CKPT = '/home/ubuntu/ObjectDetection/scripts/model_pb_generation_scripts/inception_resnet_v2_2016_08_30.ckpt'
# INCEPTIONRESNETV2_META = '/home/ubuntu/ObjectDetection/scripts/model_pb_generation_scripts/inception_resnet_v2_metaGraph.meta'

ALEXNET_CKPT = '/home/ubuntu/ObjectDetection/scripts/model_pb_generation_scripts/alexnet.ckpt'
ALEXNET_META = '/home/ubuntu/ObjectDetection/scripts/model_pb_generation_scripts/alexnet.meta'


class generic_model():
	def __init__(self, class_count):
		self.class_count = class_count

	def build_basic_graph(self, sess):
		# W_conv1 = weight_variable([8, 8, 3, 32])
		# b_conv1 = bias_variable([32])

		# W_conv2 = weight_variable([4, 4, 32, 64])
		# b_conv2 = bias_variable([64])

		# W_conv3 = weight_variable([3, 3, 64, 64])
		# b_conv3 = bias_variable([64])

		# W_fc1 = weight_variable([6400, 512])
		# b_fc1 = bias_variable([512])

		# W_fc2 = weight_variable([512, self.class_count])
		# b_fc2 = bias_variable([self.class_count])

		# imgTensor = tf.placeholder("float", [None, 80, 80, 3])
		# labelTensor = tf.placeholder("float", [None, self.class_count])
		# object_or_not = tf.placeholder("float", [None])

	 #    # hidden layers
		# conv1 = tf.nn.relu(conv2d(imgTensor, W_conv1, 4) + b_conv1)

		# conv2 = tf.nn.relu(conv2d(conv1, W_conv2, 2) + b_conv2)

		# conv3 = tf.nn.relu(conv2d(conv2, W_conv3, 1) + b_conv3)

		# conv3_flat = tf.reshape(conv3, [-1, 6400])

		# h_fc1 = tf.nn.relu(tf.matmul(conv3_flat, W_fc1) + b_fc1)

		# scores = tf.matmul(h_fc1, W_fc2) + b_fc2


		# newSaver = tf.train.import_meta_graph(ALEXNET_META)
		# newSaver.restore(sess, ALEXNET_CKPT)

		# # print [n.name for n in tf.get_default_graph().as_graph_def().node]

		# h_fc1 = sess.graph.get_tensor_by_name('fc7:0')
		# print h_fc1

		# object_or_not = tf.placeholder("float", [None])
		# labelTensor = tf.placeholder(tf.float32, [None, self.class_count])
		# imgTensor = sess.graph.get_tensor_by_name('imgTensor:0')
		# print imgTensor
		# W_fc2 = weight_variable([4096, self.class_count])
		# b_fc2 = bias_variable([self.class_count])

		# scores = tf.matmul(h_fc1, W_fc2) + b_fc2

		train_x = np.zeros((1, 227,227,3)).astype(float32)
		train_y = np.zeros((1, 1000))
		xdim = train_x.shape[1:]
		ydim = train_y.shape[1]


		net_data = load("bvlc_alexnet.npy").item()

		def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
		    '''From https://github.com/ethereon/caffe-tensorflow
		    '''
		    c_i = input.get_shape()[-1]
		    assert c_i%group==0
		    assert c_o%group==0
		    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
		    
		    
		    if group==1:
		        conv = convolve(input, kernel)
		    else:
		        input_groups = tf.split(3, group, input)
		        kernel_groups = tf.split(3, group, kernel)
		        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
		        conv = tf.concat(3, output_groups)
		    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])



		x = tf.placeholder(tf.float32, (None,) + xdim, name = 'imgTensor')

		#conv1
		#conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
		k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
		conv1W = tf.Variable(net_data["conv1"][0])
		conv1b = tf.Variable(net_data["conv1"][1])
		conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
		conv1 = tf.nn.relu(conv1_in)

		#lrn1
		#lrn(2, 2e-05, 0.75, name='norm1')
		radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
		lrn1 = tf.nn.local_response_normalization(conv1,
		                                                  depth_radius=radius,
		                                                  alpha=alpha,
		                                                  beta=beta,
		                                                  bias=bias)

		#maxpool1
		#max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
		k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
		maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


		#conv2
		#conv(5, 5, 256, 1, 1, group=2, name='conv2')
		k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
		conv2W = tf.Variable(net_data["conv2"][0])
		conv2b = tf.Variable(net_data["conv2"][1])
		conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
		conv2 = tf.nn.relu(conv2_in)


		#lrn2
		#lrn(2, 2e-05, 0.75, name='norm2')
		radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
		lrn2 = tf.nn.local_response_normalization(conv2,
		                                                  depth_radius=radius,
		                                                  alpha=alpha,
		                                                  beta=beta,
		                                                  bias=bias)

		#maxpool2
		#max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
		k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
		maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

		#conv3
		#conv(3, 3, 384, 1, 1, name='conv3')
		k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
		conv3W = tf.Variable(net_data["conv3"][0])
		conv3b = tf.Variable(net_data["conv3"][1])
		conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
		conv3 = tf.nn.relu(conv3_in)

		#conv4
		#conv(3, 3, 384, 1, 1, group=2, name='conv4')
		k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
		conv4W = tf.Variable(net_data["conv4"][0])
		conv4b = tf.Variable(net_data["conv4"][1])
		conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
		conv4 = tf.nn.relu(conv4_in)

		conv4 = tf.stop_gradient(conv4)

		#conv5
		#conv(3, 3, 256, 1, 1, group=2, name='conv5')
		k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
		conv5W = tf.Variable(net_data["conv5"][0])
		conv5b = tf.Variable(net_data["conv5"][1])
		conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
		conv5 = tf.nn.relu(conv5_in)

		#maxpool5
		#max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
		k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
		maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


		# maxpool5 = tf.stop_gradient(maxpool5)


		#fc6
		#fc(4096, name='fc6')
		fc6W = tf.Variable(net_data["fc6"][0])
		fc6b = tf.Variable(net_data["fc6"][1])
		fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b, name='fc6')

		#fc7
		#fc(4096, name='fc7')
		fc7W = tf.Variable(net_data["fc7"][0], name = 'fc7W')
		fc7b = tf.Variable(net_data["fc7"][1], name='fc7b')
		fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b, name='fc7')


		W_fc2 = weight_variable([4096, self.class_count])
		b_fc2 = bias_variable([self.class_count])
		scores = tf.matmul(fc7, W_fc2) + b_fc2

		object_or_not = tf.placeholder("float", [None])
		labelTensor = tf.placeholder("float", [None, self.class_count])

		# init = tf.initialize_all_variables()
		# sess = tf.Session()
		# sess.run(init)
		return object_or_not, labelTensor, x, scores, fc7

	def build_graph_forLwF(self, sess):
		with tf.variable_scope("AlexNet_ImageNet"):
			train_x = np.zeros((1, 227,227,3)).astype(float32)
			train_y = np.zeros((1, 1000))
			xdim = train_x.shape[1:]
			ydim = train_y.shape[1]


			net_data = load("bvlc_alexnet.npy").item()

			def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
			    '''From https://github.com/ethereon/caffe-tensorflow
			    '''
			    c_i = input.get_shape()[-1]
			    assert c_i%group==0
			    assert c_o%group==0
			    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
			    
			    
			    if group==1:
			        conv = convolve(input, kernel)
			    else:
			        input_groups = tf.split(3, group, input)
			        kernel_groups = tf.split(3, group, kernel)
			        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
			        conv = tf.concat(3, output_groups)
			    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])



			x = tf.placeholder(tf.float32, (None,) + xdim, name = 'imgTensor')

			#conv1
			#conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
			k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
			conv1W = tf.Variable(net_data["conv1"][0])
			conv1b = tf.Variable(net_data["conv1"][1])
			conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
			conv1 = tf.nn.relu(conv1_in)

			#lrn1
			#lrn(2, 2e-05, 0.75, name='norm1')
			radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
			lrn1 = tf.nn.local_response_normalization(conv1,
			                                                  depth_radius=radius,
			                                                  alpha=alpha,
			                                                  beta=beta,
			                                                  bias=bias)

			#maxpool1
			#max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
			k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
			maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


			#conv2
			#conv(5, 5, 256, 1, 1, group=2, name='conv2')
			k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
			conv2W = tf.Variable(net_data["conv2"][0])
			conv2b = tf.Variable(net_data["conv2"][1])
			conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
			conv2 = tf.nn.relu(conv2_in)


			#lrn2
			#lrn(2, 2e-05, 0.75, name='norm2')
			radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
			lrn2 = tf.nn.local_response_normalization(conv2,
			                                                  depth_radius=radius,
			                                                  alpha=alpha,
			                                                  beta=beta,
			                                                  bias=bias)

			#maxpool2
			#max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
			k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
			maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

			#conv3
			#conv(3, 3, 384, 1, 1, name='conv3')
			k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
			conv3W = tf.Variable(net_data["conv3"][0])
			conv3b = tf.Variable(net_data["conv3"][1])
			conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
			conv3 = tf.nn.relu(conv3_in)

			#conv4
			#conv(3, 3, 384, 1, 1, group=2, name='conv4')
			k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
			conv4W = tf.Variable(net_data["conv4"][0])
			conv4b = tf.Variable(net_data["conv4"][1])
			conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
			conv4 = tf.nn.relu(conv4_in)

			#conv5
			#conv(3, 3, 256, 1, 1, group=2, name='conv5')
			k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
			conv5W = tf.Variable(net_data["conv5"][0])
			conv5b = tf.Variable(net_data["conv5"][1])
			conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
			conv5 = tf.nn.relu(conv5_in)

			#maxpool5
			#max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
			k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
			maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


			# maxpool5 = tf.stop_gradient(maxpool5)


			#fc6
			#fc(4096, name='fc6')
			fc6W = tf.Variable(net_data["fc6"][0])
			fc6b = tf.Variable(net_data["fc6"][1])
			fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b, name='fc6')

			#fc7
			#fc(4096, name='fc7')
			fc7W = tf.Variable(net_data["fc7"][0], name = 'fc7W')
			fc7b = tf.Variable(net_data["fc7"][1], name='fc7b')
			fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b, name='fc7')

			fc7 = tf.stop_gradient(fc7)

			W_fc2 = weight_variable([4096, self.class_count])
			b_fc2 = bias_variable([self.class_count])
			scores = tf.matmul(fc7, W_fc2) + b_fc2

			object_or_not = tf.placeholder("float", [None])
			labelTensor = tf.placeholder("float", [None, self.class_count])

		return object_or_not, labelTensor, x, scores, fc7

	def build_graph_for_target(self, sess, labelTensor, scores, h_fc1, object_or_not, learning_rate, lamb1, lamb2, sphericalLossType, I_hfc1):

		if sphericalLossType != 'None':
			cross_entropy = tf.reduce_mean(tf.mul(tf.nn.softmax_cross_entropy_with_logits(scores, labelTensor), object_or_not))
			object_score = 0
			if sphericalLossType == 'spherical_hinge_loss':
				sphere_loss_beforeMean, norm_squared = spherical_hinge_loss(h_fc1, object_or_not)
			elif sphericalLossType == 'spherical_softmax_loss':
				sphere_loss_beforeMean, norm_squared, object_score = spherical_softmax_loss(h_fc1, object_or_not)

			sphere_loss = tf.reduce_mean(sphere_loss_beforeMean)

			kd_loss = knowledge_distillation_loss(h_fc1, I_hfc1)

			total_loss = tf.add(tf.add(cross_entropy,tf.scalar_mul(lamb1, sphere_loss)), tf.scalar_mul(lamb2, kd_loss))
			train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
			
			return cross_entropy, sphere_loss, train_step, norm_squared, object_score

		else:
			cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(scores, labelTensor))
			train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
			return cross_entropy, train_step


	
