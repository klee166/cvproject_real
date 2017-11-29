# Kwang Bin Lee
# Data Loader
# Computer Vision Project

import torch
import torchvision
import torchvision.datasets as dset
import torch.nn as nn
import random
import cv2
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


def imshow(img,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

# class MRIDataSet(Dataset):
#     def __init__(self, imageFolder, filePath, transform=None):
#         self.imageFolder = imageFolder
#         self.files = self.readf(filePath)
#         self.transform = transform

    
#     def __getitem__(self,index):

#         img1_path = os.path.join(self.imageFolder, self.files[index][0])


#         label = map(float, self.files[index][2])
#         label = torch.from_numpy(np.array(label)).float()
        
#         img1 = Image.open(img1_path[:])
#         img1 = img1.convert('L')

#         if self.transform is not None:
#             img1 = self.transform(img1)


#         return img1, label

#     def readf(self, filePath):
#         files = list()
#         with open(filePath) as f:
#             for line in f:
#                 information = line.split()
#                 files.append(information)

#         return files


#     def __len__(self):
#         return len(self.files)

folder_address = './sample/'
image_route = 'PROCESSED/MPRAGE/T88_111/'
portion = '_mpr_n4_anon_111_t88_gfc_cor_110.gif'

data_number = 1
number = 1
mri_number = 1
num_files = 3

while number < num_files + 1:
	title = 'OAS'+str(data_number)+'_000'+str(number)+'_MR'+str(mri_number)
	im_path = os.path.join(folder_address, title)
	im_path = os.path.join(im_path, image_route)
	title_image = title + portion
	im_path = os.path.join(im_path, title_image)
	img1 = Image.open(im_path[:]).convert("L")
	img1.show()	
	number = number + 1

        #return len(dset.ImageFolder(root=self.imageFolder).imgs)
