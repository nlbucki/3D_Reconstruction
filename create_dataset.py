import os, os.path
import glob
import argparse
import cv2
from matplotlib import pyplot as plt
from scipy import ndimage

parser = argparse.ArgumentParser(description='Create dataset of images and masks.')
parser.add_argument('img_folder', help='Folder containing the input images.')
args = parser.parse_args()

base_folder = args.img_folder
folders = glob.glob(os.path.join(base_folder, '*'))
for f in folders:
    if os.path.isdir(f):
		folder_name = os.path.basename(f)
		pic_list = []
		pics = glob.glob(os.path.join(f,'*.JPG'))
		for p in pics:
			img = cv2.imread(p)
			if img.shape[0] > 1000:
				img_sq = img[:,1000:5000]
			else:
				img_sq = img
			img_final = cv2.resize(img_sq, (448,448))
			cv2.imwrite(p, img_final)