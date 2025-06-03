import argparse, json, math
import scipy.io
import numpy as np
import cv2
import torch
import hdf5storage

import models.opticallayer as optical_layer
import models.resunet_vb2 as refine
from training import optical_model


def max_proj(x, axis = 0):
    return np.max(x,axis)

def mean_proj(x, axis = 0):
    return np.mean(x,axis)


def load_saved_args(model_file_path):
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--device', default='0')
    args = parser.parse_args("--device 1".split())

    with open(model_file_path+'args.json', "r") as f:
        args.__dict__=json.load(f)
    return args


def crop_images(img, image_height_org, image_width_org, image_height_crop, image_width_crop, image_shift_height, image_shift_width):
    
    crop_start_x = (image_height_org - image_height_crop) // 2 + image_shift_height
    crop_end_x = crop_start_x + image_height_crop
    crop_start_y = (image_width_org - image_width_crop) // 2 + image_shift_width
    crop_end_y = crop_start_y + image_width_crop
    
    img = img[crop_start_x:crop_end_x,crop_start_y:crop_end_y]
    
    return img


def crop_out_single(img, out_height_crop, out_width_crop, image_shift_height, image_shift_width):
    
    h,w = img.shape
    
    crop_start_x = (h - out_height_crop) // 2 - image_shift_height
    crop_end_x = crop_start_x + out_height_crop
    crop_start_y = (w - out_width_crop) // 2 - image_shift_width
    crop_end_y = crop_start_y + out_width_crop
    
    img = img[crop_start_x:crop_end_x,crop_start_y:crop_end_y]
    
    return img

def crop_out_training(img, out_height_crop, out_width_crop, image_shift_height, image_shift_width):
    
    #print(img.shape)
    a,b,h,w = img.shape
    
    crop_start_x = (h - out_height_crop) // 2 - image_shift_height
    crop_end_x = crop_start_x + out_height_crop
    crop_start_y = (w - out_width_crop) // 2 - image_shift_width
    crop_end_y = crop_start_y + out_width_crop
    
    img = img[:,:,crop_start_x:crop_end_x,crop_start_y:crop_end_y]
    
    return img
