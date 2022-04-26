"""
CS 6384 Homework 5 Programming
Implement the __getitem__() function in this python script
"""
import torch
import torch.utils.data as data
import csv
import os, math
import sys
import time
import random
import numpy as np
import cv2
import glob
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torchvision.transforms as transforms
from torchvision.io import read_image
import torchvision.transforms.functional as fn



# The dataset class
class CrackerBox(data.Dataset):
    def __init__(self, image_set = 'train', data_path = 'data'):

        self.name = 'cracker_box_' + image_set
        self.image_set = image_set
        self.data_path = data_path
        self.classes = ('__background__', 'cracker_box')
        self.width = 640
        self.height = 480
        self.yolo_image_size = 448
        self.scale_width = self.yolo_image_size / self.width
        self.scale_height = self.yolo_image_size / self.height
        self.yolo_grid_num = 7
        self.yolo_grid_size = self.yolo_image_size / self.yolo_grid_num
        # split images into training set and validation set
        self.gt_files_train, self.gt_files_val = self.list_dataset()
        # the pixel mean for normalization
        self.pixel_mean = np.array([[[102.9801, 115.9465, 122.7717]]], dtype=np.float32)

        # training set
        if image_set == 'train':
            self.size = len(self.gt_files_train)
            self.gt_paths = self.gt_files_train
            print('%d images for training' % self.size)
        else:
            # validation set
            self.size = len(self.gt_files_val)
            self.gt_paths = self.gt_files_val
            print('%d images for validation' % self.size)


    # list the ground truth annotation files
    # use the first 100 images for training
    def list_dataset(self):
        
        filename = os.path.join(self.data_path, '*.txt')
        # print(filename)
        gt_files = sorted(glob.glob(filename))
        
        gt_files_train = gt_files[:100]
        gt_files_val = gt_files[100:]
        
        return gt_files_train, gt_files_val


    # TODO: implement this function
    def __getitem__(self, idx):
    
        # print(self.gt_paths)
        filename_gt = self.gt_paths[idx]

        # gt_box_blob = torch.full((5,7,7),0.0)
        # gt_mask_blob = torch.full((7,7),0)
        # with open(filename_gt) as f:
        #     for line in f:
        #         # print(line)
        #         x1,y1,x2,y2 = map(float,line.split(" "))


        #         x1 *= 448/640
        #         x2 *= 448/640

        #         y1 *= 448/480
        #         y2 *= 448/480

        #         cx = (x1+x2)/2
        #         cy = (y1+y2)/2

        #         if x1<x2:
        #             leftx = x1
        #             lefty = y1
        #         else:
        #             leftx = x2
        #             lefty = y2

        #         # print(leftx,lefty,cx,cy)
        #         w = abs(leftx - cx)/448
        #         h = abs(lefty - cy)/448

        #         cx*=7/448
        #         cy*=7/448

        #         topleftx = math.floor(cx)
        #         toplefty = math.ceil(cy)

        #         offx = abs(topleftx - cx)/7
        #         offy = abs(toplefty-cy)/7

        #         w = abs(leftx - cx)/448
        #         h = abs(lefty - cy)/448

        #         gt_box_blob[:,topleftx,toplefty] = torch.tensor([offx,offy,w,h,1])
        #         gt_mask_blob[topleftx,toplefty]=1

        #         # break


        # fileImage = filename_gt.split("-")[0] + ".jpg"

        # image = read_image(fileImage)
        # image = fn.resize(image, size=[448,448])
        # t  = (image - self.pixel_mean.T)/255
        # image_blob = t


        # # t = torch.full((3,448,448),0.0)
        # # t[0] = image[0]-float(self.pixel_mean[0][0][0])
        # # t[2] = image[1]-float(self.pixel_mean[0][0][1])
        # # t[1] = image[2]-float(self.pixel_mean[0][0][2])
        # # t=t/255

        # # print(image,image.shape)
        # t  = (image - self.pixel_mean.T)/255
        # image_blob = t




        filename_gt = self.gt_paths[idx]
        fileImage = "-".join(filename_gt.split("-")[:-1]) + ".jpg"
        
        print(filename_gt)
        print(fileImage)
        
        # imageblob 
        image_blob = cv2.imread(fileImage)
        image_blob = cv2.resize(image_blob, (0, 0), fx=448/640, fy=448/480).astype('float32')
        image_blob = (image_blob-self.pixel_mean)/255
        image_blob = torch.FloatTensor(np.moveaxis(image_blob, 2, 0))
        
        # gt_box and gt_mask
        gt_box_blob = np.zeros((5, 7, 7))
        gt_mask_blob = np.zeros((7,7))

        with open(filename_gt) as f:
            for line in f:
                # print(line)
                x1,y1,x2,y2 = map(float,line.split(" "))
                x1,y1,x2,y2 = map(float,line.split(" "))
                x1 *= 448/640
                x2 *= 448/640

                y1 *= 448/480
                y2 *= 448/480

                cx = (x1+x2)/2
                cy = (y1+y2)/2

                w = x2-x1
                h = y2-y1


        # setting confidence pixels 1
        for i in range(7):
            for j in range(7):
                gt_box_blob[:,i,j] = ((cx-(math.floor(cx/64)*64))/64, (cy-(math.floor(cy/64)*64))/64, w/448, h/448,1)
        gt_mask_blob[math.floor(cy/64), math.floor(cx/64)] = 1
            
        gt_box_blob = torch.FloatTensor(gt_box_blob)
        gt_mask_blob = torch.FloatTensor(gt_mask_blob)
        
        # this is the sample dictionary to be returned from this function
        sample = {'image': image_blob,
                  'gt_box': gt_box_blob,
                  'gt_mask': gt_mask_blob}

        return sample


    # len of the dataset
    def __len__(self):
        return self.size
        

# draw grid on images for visualization
def draw_grid(image, line_space=64):
    H, W = image.shape[:2]
    image[0:H:line_space] = [255, 255, 0]
    image[:, 0:W:line_space] = [255, 255, 0]


# the main function for testing
if __name__ == '__main__':
    dataset_train = CrackerBox('train')
    dataset_val = CrackerBox('val')
    
    # dataloader
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=0)
    
    # visualize the training data
    for i, sample in enumerate(train_loader):
        
        image = sample['image'][0].numpy().transpose((1, 2, 0))
        gt_box = sample['gt_box'][0].numpy()
        gt_mask = sample['gt_mask'][0].numpy()

        y, x = np.where(gt_mask == 1)
        cx = gt_box[0, y, x] * dataset_train.yolo_grid_size + x * dataset_train.yolo_grid_size
        cy = gt_box[1, y, x] * dataset_train.yolo_grid_size + y * dataset_train.yolo_grid_size
        w = gt_box[2, y, x] * dataset_train.yolo_image_size
        h = gt_box[3, y, x] * dataset_train.yolo_image_size

        x1 = cx - w * 0.5
        x2 = cx + w * 0.5
        y1 = cy - h * 0.5
        y2 = cy + h * 0.5

        print(image.shape, gt_box.shape)
        
        # visualization
        fig = plt.figure()
        ax = fig.add_subplot(1, 3, 1)
        im = image * 255.0 + dataset_train.pixel_mean
        im = im.astype(np.uint8)
        plt.imshow(im[:, :, (2, 1, 0)])
        plt.title('input image (448x448)', fontsize = 16)

        ax = fig.add_subplot(1, 3, 2)
        draw_grid(im)
        plt.imshow(im[:, :, (2, 1, 0)])
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='g', facecolor="none")
        ax.add_patch(rect)
        plt.plot(cx, cy, 'ro', markersize=12)
        plt.title('Ground truth bounding box in YOLO format', fontsize=16)
        
        ax = fig.add_subplot(1, 3, 3)
        plt.imshow(gt_mask)
        plt.title('Ground truth mask in YOLO format (7x7)', fontsize=16)
        plt.show()
