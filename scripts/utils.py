import numpy as np
import os, sys
import tensorflow as tf

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

def load_dataset(BASE_PATH):
  image_list = os.listdir(BASE_PATH)
  all_image_path = []
  for image_filepath in image_list:
    all_image_path.append(os.path.join(BASE_PATH, image_filepath))
  return all_image_path
