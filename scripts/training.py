import tensorflow as tf
from genericModel import *
from utils import *
from nn_utils import weight_variable, bias_variable, conv2d, max_pool_2x2, add_last_layer
import os, sys
import numpy as np
from PIL import Image
from random import shuffle
import argparse
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


tsne_switch = False
tsne_X_array = []
tsne_colors = []

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--MAINFILE_PATH', type=str, help='The main path where the codebase exists')
	parser.add_argument('--batch_size', type=int, help='Batch size for training')
	parser.add_argument('--num_epochs', type=int, help='Number of epochs to be trained for', default = 1)
	parser.add_argument('--positiveImages_path_textfile', type=str, default = 'VOCPositiveCrops.txt')
	parser.add_argument('--negativeImages_path_textfile', type=str, default = 'VOCNegativeCrops.txt')
	parser.add_argument('--positiveImagesDirName', type = str)
	parser.add_argument('--negativeImagesDirName', type = str)
	parser.add_argument('--SAVED_NETWORKS_PATH', type = str)
	parser.add_argument('--background_fraction', type=float, default= 0.2)
	parser.add_argument('--class_count', type=int, default=21)
	args = parser.parse_args()
	ensure_dir_exists(args.SAVED_NETWORKS_PATH)
	return args

def get_all_paths_and_labels(textFilePath, ImagesDirName):
	ImagePaths = []
	ImageLabels = []
	f = open(textFilePath, 'r')

	allLines = f.readlines()
	index_shuff = range(len(allLines))
	shuffle(index_shuff)

	for i in index_shuff:
		line = allLines[i]
		line = line.split()
		ImagePaths.append(os.path.join(args.MAINFILE_PATH, ImagesDirName, line[0]))
		ImageLabels.append(int(line[1]))

	return ImagePaths, ImageLabels


args = parse_args()

##################################################################################

sess = tf.InteractiveSession()

################## Initialise a model ############################################

Model = generic_model(args.class_count)
object_or_not, labelTensor, imgTensor, scores, h_fc1 = Model.build_basic_graph(sess)
cross_entropy, sphere_loss, train_step, norm_squared = Model.build_graph_for_target(sess, labelTensor, scores, h_fc1, object_or_not)

#######################################################################################

positiveImagePaths, positiveImageLabels = get_all_paths_and_labels(args.positiveImages_path_textfile, args.positiveImagesDirName)
negativeImagePaths, negativeImageLabels = get_all_paths_and_labels(args.negativeImages_path_textfile, args.negativeImagesDirName)

#############################################################################################
saver = tf.train.Saver()

checkpoint = tf.train.get_checkpoint_state(args.SAVED_NETWORKS_PATH)
if checkpoint and checkpoint.model_checkpoint_path:
	checkpoint_IterNum = int(checkpoint.model_checkpoint_path.split('-')[-1])
	saver.restore(sess, checkpoint.model_checkpoint_path)
	print "Successfully loaded:", checkpoint.model_checkpoint_path
else:
	checkpoint_IterNum = 0
	print "Could not find old network weights"
##########################################################################################
num_positive_images = len(positiveImageLabels)
num_negative_images = len(negativeImageLabels)

positive_batch_size = int((1 - args.background_fraction)*args.batch_size)
negative_batch_size = args.batch_size - positive_batch_size

for epoch_num in range(args.num_epochs):
	
	if args.background_fraction == 1.0:
		for i in range(100):
			negative_minibatch = random.sample(range(num_negative_images), args.batch_size)		
			image_inputs = []
			label_inputs_one_hot = np.zeros((args.batch_size, args.class_count))
			object_or_not_inputs = []
			for i, index in enumerate(negative_minibatch):
				image_input = Image.open(negativeImagePaths[index]).resize((80,80))
				image_input = image_input.convert('RGB')
				image_input = np.array(image_input)
				label_index = negativeImageLabels[index]

				image_inputs.append(image_input)
				label_inputs_one_hot[i, label_index] = 1
				object_or_not_inputs.append(0)
			h_fc1_value, cross_entropy_value, sphere_loss_value, norm_squared_value = sess.run([h_fc1, cross_entropy, sphere_loss, norm_squared], feed_dict = {labelTensor: label_inputs_one_hot, imgTensor: image_inputs, object_or_not: object_or_not_inputs})
			print ' done with cross entropy loss = ' + str(cross_entropy_value) + ' and with hinge loss = ' + str(sphere_loss_value)
			print h_fc1_value
			if tsne_switch:
				tsne_X_array.append(h_fc1_value[0])


	else:
		for positive_batch_iter in range(0, num_positive_images, positive_batch_size):
			positive_start = positive_batch_iter
			positive_end = min(num_positive_images, positive_batch_iter + positive_batch_size)
			image_inputs = []
			label_inputs_one_hot = np.zeros((args.batch_size, args.class_count))
			object_or_not_inputs = []

			for image_iter in range(positive_start, positive_end):
				# image_input = Image.open(positiveImagePaths[image_iter]).resize((299, 299))
				image_input = Image.open(positiveImagePaths[image_iter]).resize((80, 80))		
				image_input = image_input.convert('RGB')
				image_input = np.array(image_input)

				label_index = positiveImageLabels[image_iter]

				image_inputs.append(image_input)
				label_inputs_one_hot[image_iter - positive_start, label_index] = 1
				object_or_not_inputs.append(1)

			negative_minibatch = random.sample(range(num_negative_images), args.batch_size - (positive_end - positive_start))
			for i, index in enumerate(negative_minibatch):
				image_input = Image.open(negativeImagePaths[index]).resize((80,80))
				image_input = image_input.convert('RGB')
				image_input = np.array(image_input)

				label_index = negativeImageLabels[index]

				image_inputs.append(image_input)
				label_inputs_one_hot[positive_end - positive_start + i, label_index] = 1
				object_or_not_inputs.append(0)



			# train_step.run(feed_dict = {labelTensor: label_inputs_one_hot, imgTensor: image_inputs, object_or_not: object_or_not_inputs})
			h_fc1_value, cross_entropy_value, sphere_loss_value, norm_squared_value = sess.run([h_fc1, cross_entropy, sphere_loss, norm_squared], feed_dict = {labelTensor: label_inputs_one_hot, imgTensor: image_inputs, object_or_not: object_or_not_inputs})
			print 'batch ' + str(positive_end) + ' done with cross entropy loss = ' + str(cross_entropy_value) + ' and with hinge loss = ' + str(sphere_loss_value)
			if tsne_switch:
				tsne_X_array.append(h_fc1_value)
				tsne_colors.append(object_or_not_inputs)


	tsne_X_array = np.array(tsne_X_array)
	tsne_X_array = np.reshape(tsne_X_array, (-1, 512))

	tsne_colors = np.array(tsne_colors)
	tsne_colors = np.reshape(tsne_colors, (-1,1))

	print tsne_colors.shape
	# tsne_X_array = np.reshape()
	tsne_model = TSNE(n_components=2, random_state=0)
	tsne_output = tsne_model.fit_transform(tsne_X_array)
	X_arr = tsne_output[:,0]
	Y_arr = tsne_output[:,1]
	# Z_arr = tsne_output[:,2]
	plt.scatter(X_arr, Y_arr, c = tsne_colors)
	plt.show()


	

	if epoch_num % 10 == 0:
		saver.save(sess, args.SAVED_NETWORKS_PATH + '/' + 'weights', global_step = (epoch_num+1)*num_positive_images + checkpoint_IterNum)
