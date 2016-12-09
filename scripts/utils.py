import numpy as np
import os, sys
import tensorflow as tf
import scipy.io
import cv2


def ensure_name_has_port(tensor_name):
  """Makes sure that there's a port number at the end of the tensor name.

  Args:
    tensor_name: A string representing the name of a tensor in a graph.

  Returns:
    The input string with a :0 appended if no port was specified.
  """
  if ':' not in tensor_name:
    name_with_port = tensor_name + ':0'
  else:
    name_with_port = tensor_name
  return name_with_port

def ensure_dir_exists(dir_name):
  """Makes sure the folder exists on disk.

  Args:
    dir_name: Path string to the folder we want to create.
  """
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)

# def load_dataset(BASE_PATH):
#   image_list = os.listdir(BASE_PATH)
#   all_image_path = []
#   for image_filepath in image_list:
#     all_image_path.append(os.path.join(BASE_PATH, image_filepath))
#   return all_image_path


def get_regions_dictionary(MAT_FILEPATH, IMAGES_BASEPATH):
  all_variables = scipy.io.loadmat(MAT_FILEPATH)
  bbox_dictionary = {}
  image_name_list = all_variables['images'][0]

  for i, image_name in enumerate(image_name_list):
    image_name = image_name[0]
    bbox_dictionary[image_name] = all_variables['boxes'][0][i]

  return bbox_dictionary

# DESKTOP = '/Users/vishakhhegde/Desktop/'
# image_dir = os.path.join(DESKTOP, 'VOCdevkit', 'VOC2012', 'JPEGImages')

# bbox_dictionary = get_regions_dictionary('/Users/vishakhhegde/ObjectDetection/selective_search_data/voc_2012_train.mat', image_dir)