import os, os.path
import glob
import argparse
import cv2
from matplotlib import pyplot as plt
from scipy import ndimage

parser = argparse.ArgumentParser(description='Renumber dataset of images and masks.')
parser.add_argument('img_folder', help='Folder containing the input images.')
args = parser.parse_args()

base_folder = args.img_folder
folders = glob.glob(os.path.join(base_folder, '*'))
for f in folders:
    if os.path.isdir(f):
		folder_name = os.path.basename(f)
		pics = glob.glob(os.path.join(f,'*.jpg'))
		for i,p in enumerate(pics):
			# print os.path.basename(p)
			os.rename(p, os.path.join(os.path.dirname(p), folder_name+'_'+str(i)+'_input.jpg'))
		pics = glob.glob(os.path.join(f,'*.png'))
		for i,p in enumerate(pics):
			# print os.path.basename(p)
			os.rename(p, os.path.join(os.path.dirname(p), folder_name+'_'+str(i)+'_mask.png'))