import tensorflow as tf
from genericModel import *
from utils import *
from nn_utils import weight_variable, bias_variable, conv2d, max_pool_2x2, add_last_layer
import os, sys
import numpy as np
from PIL import Image

# This is the training script:
# Does the following:
# 1. Load and build the inceptionResnetModel 
# 2. Train for whatever the target task is
# 3. Save the checkpoint files

# The following hard_coded variables can be passed to the network via bash script
model_order = 'inceptionResnet'
network_id = '1'
DESKTOP = '/Users/vishakhhegde/Desktop/'
MAINFILE_PATH = '/Users/vishakhhegde/ObjectDetection'
image_dir = os.path.join(MAINFILE_PATH, 'VOCdevkit', 'VOC2012', 'JPEGImages')
input_ckpt_path = os.path.join(MAINFILE_PATH, 'models', 'inception_resnet_v2_2016_08_30.ckpt')
output_ckpt_dir = os.path.join(MAINFILE_PATH, 'saved_networks')
input_graph_path = os.path.join(MAINFILE_PATH, 'models', 'inception_resnet_v2_metaGraph.meta')
output_graph_path = os.path.join(MAINFILE_PATH, 'saved_networks')
class_count = 20  # This hard coding needs to be removed

TENSOR_TO_BE_GOT = 'InceptionResnetV2/Logits/Flatten/Reshape'
GROUND_TRUTH_TENSOR_NAME = 'ground_truth'
FINAL_TENSOR_NAME = 'final_tensor'
resized_input_tensor_name = 'ResizeBilinear'

batch_size = 50
##################################################################################

# Fire up a session
sess = tf.InteractiveSession()

################## Initialise a model ############################################

Model = generic_model(model_order, network_id, image_dir, input_ckpt_path, output_ckpt_dir, input_graph_path, output_graph_path, class_count)
Model.build_basic_graph(sess)
cross_entropy = Model.build_graph_for_target(sess)

##################################################################################
# Load dataset to be fed in
image_path = load_dataset(image_dir)
num_images = len(image_path)
label_indices = np.random.randint(class_count, size = num_images)


for batch_iter in range(0, num_images, batch_size):
	start = batch_iter
	end = min(num_images, batch_iter + batch_size)
	image_inputs = []
	label_inputs_one_hot = np.zeros((end-start, class_count))

	for image_iter in range(start, end):
		image_input = Image.open(image_path[image_iter]).resize((299, 299))
		image_input = image_input.convert('RGB')
		image_input = np.array(image_input)
		label_index = label_indices[image_iter]
		image_inputs.append(image_input)
		label_inputs_one_hot[image_iter-start, label_index] = 1

	input_feed_dict = {ensure_name_has_port(GROUND_TRUTH_TENSOR_NAME): label_inputs_one_hot, ensure_name_has_port(resized_input_tensor_name): image_inputs}
	cross_entropy_value = sess.run([cross_entropy], feed_dict = input_feed_dict)
	print 'batch ' + str(batch_iter) + 'done with cross entropy loss = ' + str(cross_entropy_value)

