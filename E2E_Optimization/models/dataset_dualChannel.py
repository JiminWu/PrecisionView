import skimage.io
#import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
import torch
import numpy as np
import scipy.io
from torchvision import transforms
import models.dataaug as dataaug

class load_data(Dataset):

    def __init__(self, all_files, filepath_meas, image_height_org, image_width_org,
                             image_height_crop, image_width_crop, image_shift_height, image_shift_width):#,transform=None):

        self.all_files_gt =  all_files
        self.filepath_meas = filepath_meas
        self.image_height_org = image_height_org
        self.image_width_org = image_width_org
        self.image_height_crop = image_height_crop
        self.image_width_crop = image_width_crop
        self.image_shift_height = image_shift_height
        self.image_shift_width = image_shift_width
        

    def __len__(self):
        return len(self.all_files_gt)

    def __getitem__(self, idx):
        
        crop_start_x = (self.image_height_org - self.image_height_crop) // 2 + self.image_shift_height
        crop_end_x = self.image_height_org - (self.image_height_org - self.image_height_crop) // 2 + self.image_shift_height
        crop_start_y = (self.image_width_org - self.image_width_crop) // 2 + self.image_shift_width
        crop_end_y = self.image_width_org - (self.image_width_org - self.image_width_crop) // 2 + self.image_shift_width
        
       # print(len(self.all_files_gt))
    #    if idx > 0:
         #   print(idx)
        im_gt_1 = skimage.io.imread(self.all_files_gt[idx])
        sample_1 = {'im_gt': im_gt_1.astype('float32')/255.}
        im_gt_1 = sample_1['im_gt']
        im_gt_1= im_gt_1[crop_start_x:crop_end_x,crop_start_y:crop_end_y]
            
        idx_2 = np.random.randint(1, len(self.all_files_gt))
        im_gt_2 = skimage.io.imread(self.all_files_gt[idx_2])
        sample_2 = {'im_gt_2': im_gt_2.astype('float32')/255.}
        im_gt_2 = sample_2['im_gt_2']
        im_gt_2= im_gt_2[crop_start_x:crop_end_x,crop_start_y:crop_end_y]

       # h,w = im_gt.shape
        #print(h,w)
        im_gt = np.stack((im_gt_1, im_gt_2))
      #  print(im_gt.shape)
        #im_gt = dataaug.random_rotation(im_gt, angle_range=(-50, 50))
        
        #crop_size = (self.image_height_crop, self.image_width_crop)
        #im_gt = dataaug.random_resized_crop(im_gt, crop_size)
       
        
        #print(im_gt.shape)
        return {'im_gt': torch.from_numpy(im_gt)} #,
                #'meas': torch.from_numpy(meas)}
