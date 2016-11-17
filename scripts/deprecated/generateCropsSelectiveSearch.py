import numpy as np
import selectivesearch
from PIL import Image
import os, sys
from utils import *
import cv2

def get_regions(img, scale, sigma, min_size):
	img_lbl, regions = selectivesearch.selective_search(img, scale=scale, sigma=sigma, min_size=min_size)
	candidates = set()
	for r in regions:
		# excluding same rectangle (with different segments)
		if r['rect'] in candidates:
			continue
		# excluding regions smaller than 2000 pixels
		if r['size'] < 500:
			continue
		# distorted rects
		x, y, w, h = r['rect']
		if w / h > 1.2 or h / w > 1.2:
			continue
		candidates.add(r['rect'])
	return candidates

def get_image_crops(img, candidates, image_name):
	all_crops = []
	for i, cand in enumerate(candidates):
		x,y,w,h = cand
		cand_crop = img[x:x+w, y:y+h]
		cv2.imwrite('../cropped_images/' + image_name.split('.')[0] + str(i) + '.jpg', cand_crop)
		print 'wrote file'


DESKTOP = '/Users/vishakhhegde/Desktop/'
image_dir = os.path.join(DESKTOP, 'VOCdevkit', 'VOC2012', 'JPEGImages')

image_path = load_dataset(image_dir)

scale=2000
sigma=0.9
min_size=100

for image in image_path:
	img = cv2.imread(image)
	head, image_name = os.path.split(image)
	candidates = get_regions(img, scale, sigma, min_size)
	get_image_crops(img, candidates, image_name)
	
