# Kwang Bin Lee
# Data Loader
# Computer Vision Project

import torch
import math
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


logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

parser = argparse.ArgumentParser(description="CV assignment")
parser.add_argument("--load", default=None, required=False,
                    help="Train and load a weight data ")
parser.add_argument("--save", default=None, required=False,
                    help="Train and save a weight data ")
parser.add_argument("--write", default=None, required=False, type=int,
                    help="Write a train file")
parser.add_argument("--cuda", default=0, type=int, required=True,
                    help="Set if it is cuda or not (0 = not cuda)(1 = cuda)")
parser.add_argument("--epoch", default=10, type=int,
                    help="Set the number of epochs (integer) default = 10")

args = parser.parse_args()


def imshow(img,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

class MRIDataSet(Dataset):
    def __init__(self, filePaths, transform=None):
        self.transform = transform
        self.filePaths = filePaths

    
    def __getitem__(self,index):

        img1_path = self.filePaths[index][0]


        print self.filePaths[index][1]
        label = map(float, self.filePaths[index][1])
        label = torch.from_numpy(np.array(label)).float()
        
        img1 = Image.open(img1_path[:])
        img1 = img1.convert('L')

        if self.transform is not None:
            img1 = self.transform(img1)


        return img1, label

    # def readf(self, filePath):
    #     files = list()
    #     with open(filePath) as f:
    #         for line in f:
    #             information = line.split()
    #             files.append(information)

    #     return files


    def __len__(self):
        return len(self.filePaths)


class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.cnn1 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=(5,5), stride=(1,1), padding=2),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(2, stride=(2,2)),
        )

        self.cnn2 = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=(5,5), stride=(1,1), padding=2),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(2, stride=(2,2)),
        )
        self.cnn3 = nn.Sequential(
        nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1), padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(256),
        nn.MaxPool2d(2,stride=(2,2)),
        )
        self.cnn4 = nn.Sequential(
        nn.Conv2d(256,512,kernel_size=(3,3), stride=(1,1), padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(512)
        )
        
        
        self.fc1 = nn.Sequential(
            nn.Linear(131072, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1024)

            # make output layer sine function
            #nn.Sigmoid()
        )  

        self.final = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
            )


    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

    def forward_once(self, x):
        output = self.cnn1(x)
        output = self.cnn2(output)
        output = self.cnn3(output)
        output = self.cnn4(output)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def produce_result(self, x):
        output = self.final(x)
        return output






folder_address = './sample/'
image_route = 'PROCESSED/MPRAGE/T88_111/'
portion = '_mpr_n4_anon_111_t88_gfc_cor_110.gif'
    
train_address = './train.txt'
test_address = './test.txt'

data_number = 1
number = 1
mri_number = 1
num_files = 3

train_paths = list()


if args.write is 1:

	f = open("train.txt", "w+")

	while number < num_files + 1:
		title = 'OAS'+str(data_number)+'_000'+str(number)+'_MR'+str(mri_number)
		title_txt = title+'.txt'
		im_path = os.path.join(folder_address, title)
		txt_path = os.path.join(im_path,title_txt)

		#extract data from the given text file
		f_txt = open(os.path.expanduser(txt_path))

		#extract CDR 
		txt_data = f_txt.readlines()

		cdr = txt_data[6].rstrip().split()[1]


		im_path = os.path.join(im_path, image_route)
		title_image = title + portion
		im_path = os.path.join(im_path, title_image)
		
		#img1 = Image.open(im_path[:]).convert("L")

		cdr = float(cdr)

		if cdr == 1:
			cdr = 2

		if cdr == 2:
			cdr = 3

		if cdr == 3:
			cdr = 4


		cdr = round(cdr)


		f.write("%s %d\n" %(im_path[:], cdr))
		# img1.show()	
		f_txt.close()

		number = number + 1
	f_txt.close()
	f.close()


f_train = open("train.txt", "r")

for line in f_train:
	info = line.split()
	train_paths.append(info)


train_dataset = MRIDataSet(train_paths,
                            transform=transforms.Compose([
                                                        # Resize(),
                                                        # transforms.CenterCrop((128, 128)),
                                                        # RandomHorizontalFlip(),
                                                        # RandomRotation(),
                                                        # Translate(),
                                                        transforms.Scale((128,128)),
                                                        transforms.ToTensor()
                                                        ]))

train_dl = DataLoader(train_dataset,
                    shuffle=True,
                    num_workers=2,
                    batch_size=8)


if args.cuda is 1:
    s_net = SiameseNet().cuda()
else: 
    s_net = SiameseNet()

if args.save is not None:

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(s_net.parameters(),betas=(0.9,0.9),lr=1e-3)


    for epoch in range(0, args.epoch):
        for index, data in enumerate(train_dl, 0):

            img0, label = data
            if args.cuda is 1:
                img0, label = Variable(img0).cuda(), Variable(label).cuda()
            else:
                img0, label = Variable(img0), Variable(label)

            optimizer.zero_grad()

            ov = s_net.produce_result(img0)

            bceloss_r = criterion(ov.double(), label.double())
            bceloss_r.backward()
            optimizer.step()

            if index % 10 == 0:
                print "Epoch %d Batch %d Loss %f" %(epoch, index, bceloss_r.data[0])

    torch.save(s_net.state_dict(), './'.join(args.save))

