# Kwang Bin Lee
# Data Loader
# Computer Vision Project
# MRI Data Loading

import torch
import math
import torchvision
import torchvision.datasets as dset
import torch.nn as nn
import random
#import cv2
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torch.autograd import Variable
import logging
import argparse
from shutil import copyfile


logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

parser = argparse.ArgumentParser(description="CV assignment")
parser.add_argument("--write", default=None, required=False, type=int,
                    help="Write a train file")

args = parser.parse_args()



a_address = './alzheimers/'
folder_address = './sample/'

image_route = 'PROCESSED/MPRAGE/T88_111/'
portion = '_mpr_n4_anon_111_t88_gfc_cor_110.gif'
new_portion = '_mpr_n4_anon_111_t88_gfc_cor_110.jpg'
    
train_address = './train.txt'
test_address = './test.txt'

data_number = 1
number = 1
mri_number = 1
num_files = 3

train_paths = list()


num_files = len([i for i in os.listdir(folder_address) if os.path.isdir(folder_address)])

if not os.path.exists(a_address):
	os.makedirs(a_address)

if args.write is 1:

	while number < num_files + 1:
		title = 'OAS'+str(data_number)+'_000'+str(number)+'_MR'+str(mri_number)
		im_path = os.path.join(folder_address, title)
		title_txt = title + '.txt'
		txt_path = os.path.join(im_path,title_txt)


		im_path = os.path.join(im_path, image_route)
		title_image = title + portion

		# where the image actually is
		im_path = os.path.join(im_path, title_image)

		if os.path.exists(txt_path):
			f_txt = open(os.path.expanduser(txt_path))
			txt_data = f_txt.readlines()

			if(len(txt_data[6].split()) > 1):
				cdr = txt_data[6].rstrip().split()[1]

			
				if cdr == 1:
					cdr = 2
				elif cdr == 2:
					cdr = 3
				elif cdr == 3:
					cdr = 4
				else:
					pass
				cdr = int(round(float(cdr)))
				print cdr

				new_path = './label_' + str(int(cdr))
				new_label_folder = os.path.join(a_address, new_path)

				if not os.path.exists(new_label_folder):
					os.makedirs(new_label_folder)


				new_jpg_file = os.path.join(new_label_folder, title + new_portion)
				print (new_jpg_file)
				#copyfile(im_path, new_label_file)

				im = Image.open(im_path)
				im.save(new_jpg_file,'JPEG')




		# img1.show()	

		number = number + 1
	f_txt.close()