import numpy as np
import torch, torch.optim
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor
import os, sys, json, glob
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import argparse
from PIL import Image
import hdf5storage

from torch.utils.data import Dataset, DataLoader
import cv2
import random

import models.ensemble as ensemble
import models.dataset as ds

import helper as hp
import models.loss_L1L2 as L
from models.resunet_vb2 import ResUnet_VB
import models.save_loss as SL

import time

parser = argparse.ArgumentParser()
# Training settings
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--batch_size', default=20, type=int)
parser.add_argument('--device', default='0,1')
parser.add_argument('--load_path',default=None)
#parser.add_argument('--load_path',default='./saved_data/trained_range500_resunet_L2L1_51/')#default=None
parser.add_argument('--save_checkponts',default=True)

# Loss functions
parser.add_argument('--lambda_L1',default=0.2)
parser.add_argument('--lambda_DL1',default=0.0) # 0.5*lambda_L1
parser.add_argument('--lambda_L2',default=1.0)
parser.add_argument('--lambda_DL2',default=0.0) # 0.5*lambda_L2
parser.add_argument('--lambda_RMS',default=0.0)
parser.add_argument('--lambda_SSIM',default=0.0)


def main():

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    # Image cropping parameters
    image_height_org = 368 
    image_width_org = 480 
    image_height_crop = 368 
    image_width_crop = 480 
    image_shift_height = 0
    image_shift_width = 0

    # Target reconstruction parameters
    target_height = image_height_crop 
    target_width = image_width_crop 

    filepath_gt = './Training_Data/train_groundtruth/'
    filepath_meas = './Training_Data/train_capture/'
    filepath_val_gt = './Training_Data/val_groundtruth/'
    filepath_val_meas = './Training_Data/val_capture/' 

    filepath_gt= glob.glob(filepath_gt+'*')
    filepath_meas= glob.glob(filepath_meas+'*')
    filepath_val_gt= glob.glob(filepath_val_gt+'*')
    filepath_val_meas= glob.glob(filepath_val_meas+'*')

    print('training images:', len(filepath_gt),
            'testing images:', len(filepath_val_gt))

    dataset_train = ds.load_data(filepath_gt, filepath_meas, image_height_org, image_width_org,
                                image_height_crop, image_width_crop, image_shift_height, image_shift_width)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=1)

    dataset_test = ds.load_data(filepath_val_gt, filepath_val_meas, image_height_org, image_width_org,
                                image_height_crop, image_width_crop, image_shift_height, image_shift_width)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=1)

    unet_layer = ResUnet_VB(channels=1, dim=16, out_dim=1, dim_mults=(1,2,4,8), resnet_block_groups=8)

    device = 'cuda:0'
    Gloss = L.GLoss(args).to(device)
    model = ensemble.MyEnsemble(unet_layer)        
    model.to(device)
    

    if args.load_path is not None:
        model.load_state_dict(torch.load(args.load_path + 'model_noval.pt', map_location=torch.device(device)))
        best_loss_path = args.load_path + 'best_loss.txt'
        hist_loss_path = args.load_path + 'loss_history.txt'
        best_loss = hp.get_loss_from_file(best_loss_path)
        best_loss_info = hp.get_best_loss_info_from_file(best_loss_path)

        hist_loss_info = hp.get_best_loss_info_from_file(hist_loss_path)
        epoch_number = hp.extract_epoch_number(hist_loss_info)
        args.load_path = None
        print('loading saved model', f"starting from epoch {epoch_number}", f"Best loss info: {best_loss_info}")
    else:
        best_loss = 1e6


    print("Start training")
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    
    if args.save_checkponts == True:
        filepath_save = 'saved_data/' +"trained_range500_resunet_L2L1_51/"

        if not os.path.exists(filepath_save):
            os.makedirs(filepath_save)
    
        with open(filepath_save + 'args.json', 'w') as fp:
            json.dump(vars(args), fp)
    

    try:
        starter = epoch_number+1
    except NameError:
        starter = 0

    for itr in range(starter,args.epochs):

  #      if itr % 1000 == 0 and itr > 1:
  #              args.lr = args.lr / 2
  #              for g in optimizer.param_groups:
  #                  g['lr'] = args.lr
  #                  print('Update LR: ', g['lr'])
        
        start = time.time()
        for i_batch, sample_batched in enumerate(dataloader_train):
            
            optimizer.zero_grad()

            out = model(sample_batched['meas'].to(device))
           # out = hp.crop_out_training(out, target_height, target_width, image_shift_height, image_shift_width)
            gt = sample_batched['im_gt'].to(device)
           # gt = hp.crop_out_training(gt, target_height, target_width, image_shift_height, image_shift_width)
            Gloss(output=out, target=gt)
            Gloss.total_loss.backward()
                
            
            optimizer.step()
            
            print('epoch: ', itr, 'LR:', args.lr, ' batch: ', i_batch, ' loss: ', Gloss.total_loss.item(), end='\r')


        if args.save_checkponts == True:
            torch.save(model.state_dict(), filepath_save + 'model_noval.pt')
            if itr%10==0:
                    modelname = f'model_epoch_{itr}.pt'
                    modelpath = os.path.join(filepath_save, modelname)
                    torch.save(model.state_dict(), modelpath)
        
        
        if itr%1==0:
            total_loss=0
            batch_num = 0
            for i_batch, sample_batched in enumerate(dataloader_test):
                with torch.no_grad():
                    out = model(sample_batched['meas'].to(device))
                  #  out = hp.crop_out_training(out, target_height, target_width, image_shift_height, image_shift_width)
                    gt = sample_batched['im_gt'].to(device)
                  #  gt = hp.crop_out_training(gt, target_height, target_width, image_shift_height, image_shift_width)
              
                    Gloss(output=out, target=gt)
                    total_loss+=Gloss.total_loss.item()
                    batch_num = batch_num + 1
                    
                  #  print('loss for testing image ',itr,' ',i_batch, Gloss.total_loss.item())
            avg_loss = total_loss/batch_num

            print('Total loss for testing set ',itr,' ', total_loss, 'average loss ', avg_loss, end='\n')
        
            output_numpy = out.detach().cpu().numpy()[0][0]
            gt_numpy = gt.detach().cpu().numpy()[0][0]
            meas_numpy = sample_batched['meas'].detach().cpu().numpy()[0][0]    
                 
            if args.save_checkponts == True:

                im_gt = Image.fromarray((np.clip(gt_numpy/np.max(gt_numpy),0,1)*255).astype(np.uint8))
                im = Image.fromarray((np.clip(output_numpy/np.max(output_numpy),0,1)*255).astype(np.uint8))
                im.save(filepath_save + str(itr) + str() + '.png')
                im_gt.save(filepath_save + str(itr) + 'gt.png')
                SL.save_loss_to_file(itr, avg_loss, filename = filepath_save + 'loss_history.txt')
            
            
            if avg_loss<best_loss:
                best_loss=avg_loss
    
                # update best model
                if args.save_checkponts == True:
                    torch.save(model.state_dict(), filepath_save + 'model.pt')
                    SL.save_loss_to_file(itr, avg_loss, filename = filepath_save + 'best_loss.txt')            

        stop = time.time()
        print('Total time for epoch #', itr, ': ', stop-start,'s')

        
if __name__ == '__main__':
    main()