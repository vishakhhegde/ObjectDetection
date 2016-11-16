import tensorflow as tf
import numpy as np
import os, sys
from tensorflow.python.platform import gfile

BOTTLENECK_TENSOR_NAME = 'InceptionResnetV2/Logits/Flatten/Reshape:0'


parent_directory_model = '/home/ubuntu/matroid/scripts/retrain_files/model_pb_generation_scripts/'

modelName = 'inception_resnet_v2_metaGraph.meta'
checkpoint_file = 'inception_resnet_v2_2016_08_30.ckpt'

metaFilePath = os.path.join(parent_directory_model, modelName)
checkpointFilePath = os.path.join(parent_directory_model, checkpoint_file)

sess = tf.Session()
'''
with gfile.FastGFile(graph_path, 'rb') as f:
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())
	_ = tf.import_graph_def(graph_def, name='')
'''
#saver = tf.train.Saver()
#saver.restore(sess, checkpoint_file)
#print tf.trainable_variables()

new_saver = tf.train.import_meta_graph(metaFilePath)
new_saver.restore(sess, checkpointFilePath)