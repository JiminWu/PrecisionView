import argparse, json, math
import scipy.io
import numpy as np
import cv2
import torch
import hdf5storage
import models.ensemble as ensemble
import torch.nn.functional as F
from models.resunet_vb2 import ResUnet_VB
import re


def max_proj(x, axis = 0):
    return np.max(x,axis)

def mean_proj(x, axis = 0):
    return np.mean(x,axis)


def load_saved_args(model_file_path):
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    #parser.add_argument('--psf_num', default=5, type=int)
    parser.add_argument('--device', default='0')
    args = parser.parse_args("--device 1".split())

    with open(model_file_path+'args.json', "r") as f:
        args.__dict__=json.load(f)
    return args


def load_model(model_filepath, device = 'cuda:0', load_model = True):

    args = load_saved_args(model_filepath)
    unet_layer = ResUnet_VB(channels=1, dim=16, out_dim=1, dim_mults=(1,2,4,8), resnet_block_groups=8).to(device)
    #unet_layer = Unet(n_channel_in=1, n_channel_out=1, residual=False, down='maxpool', up='bilinear', activation='relu').to(device)

    model=ensemble.MyEnsemble(unet_layer)
    
    if load_model == True:
        model.load_state_dict(torch.load(model_filepath+'model.pt',map_location=torch.device(device)))
        
    return model, args


def crop_psfs(psfs, psf_height_org, psf_width_org, psf_height_crop, psf_width_crop, shift_height, shift_width):
    
    crop_start_x = (psf_height_org - psf_height_crop) // 2 + shift_height
    crop_end_x = psf_height_org - (psf_height_org - psf_height_crop) // 2 + shift_height
    crop_start_y = (psf_width_org - psf_width_crop) // 2 + shift_width
    crop_end_y = psf_width_org - (psf_width_org - psf_width_crop) // 2 + shift_width
    
    psfs = psfs[crop_start_x:crop_end_x,crop_start_y:crop_end_y]
    psfs = psfs.transpose(2,0,1)
    
    return psfs

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
    a1,a2,h,w = img.shape
    
    crop_start_x = (h - out_height_crop) // 2 - image_shift_height
    crop_end_x = crop_start_x + out_height_crop
    crop_start_y = (w - out_width_crop) // 2 - image_shift_width
    crop_end_y = crop_start_y + out_width_crop
    
    img = img[:,:,crop_start_x:crop_end_x,crop_start_y:crop_end_y]
    
    return img

def padding(meas, padding_x, padding_y):
        
    a,b,h,w = meas.shape
    #meas = meas.type(torch.complex64)
     #   padding_x = 128
     #   padding_y = 128
    # print(h,w)
        
    if padding_x == 0 & padding_y == 0:
        img_pad = meas
        center_x = h // 2
        center_y = w // 2
     
    else:
        padding = (padding_x, padding_x, padding_y, padding_y)
        center_x = (h + 2 * padding_x) // 2
        center_y = (w + 2 * padding_y) // 2
        
        img_pad = F.pad(meas, padding, mode='replicate')
        print(img_pad.shape)
        meas = img_pad
    
    return meas

def get_loss_from_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        
    # Split the content by whitespace to get individual words/numbers
    parts = content.split()
    
    # Reverse the list and find the first number
    for part in reversed(parts):
        try:
            # Try to convert to a number
            number = float(part)
            return number
        except ValueError:
            # If it's not a number, continue
            continue
    
    # If no number is found, return None
    return None

def get_best_loss_info_from_file(file_path):
    with open(file_path, 'r') as file:
        # Read all lines into a list
        lines = file.readlines()
        
    # Check if there are any lines in the file
    if lines:
        # Return the last line (with any trailing newline character stripped)
        return lines[-1].strip()
    else:
        return None
    
def extract_epoch_number(last_row):
    # Regular expression to find the number after 'epoch #'
    match = re.search(r'epoch #(\d+)', last_row)
    if match:
        return int(match.group(1))
    else:
        return None