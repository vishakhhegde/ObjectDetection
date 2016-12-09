import numpy as np
import os, sys
import tensorflow as tf
from utils import *
import xmltodict
from time import sleep
from class_dictionary import class_dictionary
import argparse
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--MAINFILE_PATH', type=str, help='The main path where the codebase exists')
	parser.add_argument('--image_dir', type=str, help='Path to the image directory')
	parser.add_argument('--negative_cropped_image_dir', type=str, help='Path to directory which stores negative images')
	parser.add_argument('--positive_cropped_image_dir', type=str, help='Path to directory which stores positive images')
	parser.add_argument('--positiveImages_path_textfile', type=str, default = 'VOCPositiveCrops.txt')
	parser.add_argument('--negativeImages_path_textfile', type=str, default = 'VOCNegativeCrops.txt')
	parser.add_argument('--xmlFiles_dir', type=str, help='Path to the required xml file')
	parser.add_argument('--selectiveSearchFile', type=str, help='Path to file containing selective search data')
	args = parser.parse_args()

	ensure_dir_exists(args.negative_cropped_image_dir)
	ensure_dir_exists(args.positive_cropped_image_dir)
	return args

def get_bbox_max_and_min(origBbox):
	orig_xmax = int(origBbox['bndbox']['xmax'])
	orig_xmin = int(origBbox['bndbox']['xmin'])
	orig_ymax = int(origBbox['bndbox']['ymax'])
	orig_ymin = int(origBbox['bndbox']['ymin'])
	# This is fixed
	orig_bbox = (orig_ymin, orig_xmin, orig_ymax, orig_xmax)
	return orig_bbox

def get_orig_bbox(xml_filepath):
	xml_file = open(xml_filepath, 'r')
	objDict_list = xmltodict.parse(xml_file)['annotation']['object']
	if not type(objDict_list) is list:
		objDict_list = [objDict_list]
	return objDict_list

def find_IoU(box1, box2):
	# This is fixed
	inter_ymin = max(box1[0], box2[0])
	inter_xmin = max(box1[1], box2[1])
	inter_ymax = min(box1[2], box2[2])
	inter_xmax = min(box1[3], box2[3])

	check1 = inter_xmax - inter_xmin
	check2 = inter_ymax - inter_ymin
	if check1 < 0 or check2 < 0:
		intersection = 0
		return 0
	else:
		intersection = (inter_xmax - inter_xmin + 1)*(inter_ymax - inter_ymin + 1)
		box1_area = (box1[3] - box1[1] + 1)*(box1[2] - box1[0] + 1)
		box2_area = (box2[3] - box2[1] + 1)*(box2[2] - box2[0] + 1)
		union = box1_area + box2_area - intersection
		IoU = float(intersection)/float(union)
		return IoU

def is_object(bbox, origBbox_list, IoU_threshold = 0.7):
	# This is fixed
	ymin, xmin, ymax, xmax = bbox
	# This is fixed
	bbox = (int(ymin), int(xmin), int(ymax), int(xmax))
	object_switch = 0
	
	nameList = []
	IoUList = []

	for origBbox in origBbox_list:
		orig_bbox = get_bbox_max_and_min(origBbox)
		IoU = find_IoU(orig_bbox, bbox)
		IoUList.append(IoU)
		nameList.append(origBbox['name'])

	IoU = max(IoUList)
	objectName = nameList[IoUList.index(IoU)]

	if IoU > IoU_threshold:
		object_switch = 1
		return objectName, object_switch

	return 'background', object_switch

def crop_single_image(orig_img, bbox):
	ymin, xmin, ymax, xmax = bbox
	cropped_img = orig_img[ymin:ymax+1, xmin:xmax+1]
	return cropped_img

def crop_all_images_in_dict(bbox_dictionary, image_dir, positive_cropped_image_dir, negative_cropped_image_dir, xmlFiles_dir):

	for image_name in bbox_dictionary:
		xml_filepath = os.path.join(xmlFiles_dir, image_name + '.xml')
		origBbox_list = get_orig_bbox(xml_filepath)
		
		image_path = os.path.join(image_dir, image_name + '.jpg')

		orig_img = Image.open(image_path)
		orig_img = orig_img.convert('RGB')
		orig_img = np.array(orig_img)

		# orig_img = cv2.imread(image_path)

		for j, origBbox in enumerate(origBbox_list):
			orig_bbox = get_bbox_max_and_min(origBbox)
			cropped_img = crop_single_image(orig_img, orig_bbox)
			cropped_img_name = image_name + '_orig_' + str(j) + '.jpg'
			cropped_img = Image.fromarray(cropped_img, mode='RGB')
			cropped_img.save(os.path.join(positive_cropped_image_dir, cropped_img_name))

			# cv2.imwrite(os.path.join(positive_cropped_image_dir, cropped_img_name), cropped_img)

			object_class = origBbox['name']

			lineToWrite = cropped_img_name + '\t' + str(class_dictionary[object_class]) + '\n'
			f_positiveImages.write(lineToWrite)


		for i, bbox in enumerate(bbox_dictionary[image_name]):
			object_class, object_switch = is_object(bbox, origBbox_list)			
			cropped_img_name = image_name + '_' + str(i) + '.jpg'

			print object_class, class_dictionary[object_class]

			# Now we actually need to crop the image
			cropped_img = crop_single_image(orig_img, bbox)
			lineToWrite = cropped_img_name + '\t' + str(class_dictionary[object_class]) + '\n'
			cropped_img = Image.fromarray(cropped_img, mode='RGB')

			if object_switch:
				# cv2.imwrite(os.path.join(positive_cropped_image_dir, cropped_img_name), cropped_img)
				cropped_img.save(os.path.join(positive_cropped_image_dir, cropped_img_name))
				f_positiveImages.write(lineToWrite)
			else:
				# cv2.imwrite(os.path.join(negative_cropped_image_dir, cropped_img_name), cropped_img)
				cropped_img.save(os.path.join(negative_cropped_image_dir, cropped_img_name))
				f_negativeImages.write(lineToWrite)

#######################################################################################
def main(args):
	# Get a dictionary of bounding boxes
	bbox_dictionary = get_regions_dictionary(args.selectiveSearchFile, args.image_dir)
	# Crop all images
	crop_all_images_in_dict(bbox_dictionary, args.image_dir, args.positive_cropped_image_dir, args.negative_cropped_image_dir, args.xmlFiles_dir)

######################################################################################

if __name__ == "__main__":
	args = parse_args()
	f_positiveImages = open(args.positiveImages_path_textfile, 'w')
	f_negativeImages = open(args.negativeImages_path_textfile, 'w')
	main(args)
	print 'Finished obtaining region proposals for all images'


