import numpy as np
import torch, torch.optim
import torch.nn.functional as F
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor
import os, sys, json, glob
import torchvision
from torchvision import transforms
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from math import ceil
import argparse
from PIL import Image
import hdf5storage
import time

from torch.utils.data import Dataset, DataLoader
import cv2

import models.opticallayer as optical_layer
import models.dataset as ds
import helper as hp
import models.loss as L
import models.resunet_vb2 as refine
import models.dataaug as dataaug
import models.save_loss as SL
from models_unets.unet import Unet

parser = argparse.ArgumentParser()
# Training settings
parser.add_argument('--epochs', default=3000, type=int)
parser.add_argument('--digital_lr', default=0, type=float)  #1e-4
parser.add_argument('--optical_lr', default=0, type=float)  #1e-9
parser.add_argument('--batch_size', default=20, type=int) #21 for range 600
parser.add_argument('--device', default='0')
#parser.add_argument('--load_path_optical',default='./saved_data/trained_448resunet_0607/optical_model.pt')#default=None
#parser.add_argument('--load_path_unet',default='./saved_data/trained_448resunet_0607/unet_model.pt')#default=None
parser.add_argument('--load_path_optical',default=None)#default=None
parser.add_argument('--load_path_unet',default=None)#default=None
parser.add_argument('--save_checkponts',default=True)

# Loss functions
parser.add_argument('--lambda_SSIM',default=0.5)
parser.add_argument('--lambda_L1',default=0.0)
parser.add_argument('--lambda_RMS',default=1.5)


device = 'cuda:0'
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.device


target_height = 448
target_width = 448

# Image cropping parameters
image_height_org = 512
image_width_org = 512
image_height_crop = target_height+64
image_width_crop = target_width+64
image_shift_height = 0
image_shift_width = 0

# Target reconstruction parameters


######################################### Set parameters   ###############################################

zernike = hdf5storage.loadmat('zernike_basis_small_25.mat')
u2 = zernike['u2']  # basis of zernike poly
idx = zernike['idx'] # area mask
idx = idx.astype(np.float32)

a_zernike_mat = hdf5storage.loadmat('a_zernike_cubic_150mm.mat')
a_zernike_fix = a_zernike_mat['a']
a_zernike_fix = a_zernike_fix * 4
a_zernike_fix = torch.tensor(a_zernike_fix)

N_B = 25  # size of the blur kernel
wvls = np.array([530]) * 1e-9 # wavelength 550 nm
N_color = len(wvls)

N_modes = u2.shape[1]  # load zernike modes

# generate the defocus phase
N_Phi = 20
Phi_list = np.linspace(-19, 19, N_Phi, np.float32) # defocus (kwm)


# baseline offset for the heightmap
c = 0

########################################### DATA PATH ################################################
filepath_gt = 'C:/Users/Huayu/Desktop/Huayu/Projects/HRME-LFOV/finetune/Data/E2E/train_gt/' #processed_gt_2048/'
filepath_val_gt = 'C:/Users/Huayu/Desktop/Huayu/Projects/HRME-LFOV/finetune/Data/E2E/val_gt/' #processed_val_gt/'


filepath_gt= glob.glob(filepath_gt+'*')
filepath_val_gt= glob.glob(filepath_val_gt+'*')

print('training images:', len(filepath_gt),
          'testing images:', len(filepath_val_gt))


dataset_train = ds.load_data(filepath_gt, filepath_gt, image_height_org, image_width_org,
                             image_height_crop, image_width_crop, image_shift_height, image_shift_width)
dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=1)

dataset_test = ds.load_data(filepath_val_gt, filepath_gt, image_height_org, image_width_org,
                             image_height_crop, image_width_crop, image_shift_height, image_shift_width)
dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=1)


unet_layer = refine.ResUnet_VB(channels=1, dim=16, out_dim=1, dim_mults=(1,2,4,8), resnet_block_groups=8).to(device)
#unet_layer = Unet(n_channel_in=1, n_channel_out=1, residual=False, down='maxpool', up='nearest', activation='relu').to(device)

gamma = 1
a_zernike_learn = torch.zeros_like(a_zernike_fix)
optical = optical_layer.fft_conv(gamma, a_zernike_learn, Phi_list=Phi_list, N_B=N_B, wvls=wvls, a_zernike_fix=a_zernike_fix, idx=idx, u2=u2).to(device)
optical_model = optical_layer.OpticalEnsemble(optical)
unet_model = refine.UnetEnsemble(unet_layer)


Gloss = L.GLoss(args).to(device)




if __name__ == '__main__':

    optical_optimizer = torch.optim.Adam(optical_model.parameters(), lr = args.optical_lr)
    digital_optimizer = torch.optim.Adam(unet_model.parameters(), lr = args.digital_lr)
#    Doptimizer = torch.optim.Adam(D_model.parameters(), lr = args.lr)
    
    if args.load_path_optical is not None:
        optical_model.load_state_dict(torch.load(args.load_path_optical, map_location=torch.device(device)))
   # unet_model.load_state_dict(torch.load(args.load_path_unet, map_location=torch.device(device)))
        print('loading saved optical model')


    if args.load_path_unet is not None:
   # optical_model.load_state_dict(torch.load(args.load_path_optical, map_location=torch.device(device)))
        unet_model.load_state_dict(torch.load(args.load_path_unet, map_location=torch.device(device)))
        print('loading saved unet model')


    if args.save_checkponts == True:
        filepath_save = 'saved_data/' +"trained_saveinitialization/"
    
        if not os.path.exists(filepath_save):
            os.makedirs(filepath_save)
    
        with open(filepath_save + 'args.json', 'w') as fp:
            json.dump(vars(args), fp)
    
    best_loss=1e6
    
    
    for itr in range(0,args.epochs):
        
 #       if itr % 1000 == 0 and itr > 1:
 #               args.lr = args.lr / 2
 #               for g in optimizer.param_groups:
 #                   g['lr'] = args.lr
 #                   print('Update LR: ', g['lr'])

        start = time.time()

        for i_batch, sample_batched in enumerate(dataloader_train):
            optical_optimizer.zero_grad()
            digital_optimizer.zero_grad()
          #  Doptimizer.zero_grad()
            
            [optical_out,a_zernike_out, h_out, PSFs_out] = optical_model(sample_batched['im_gt'].to(device))
           # print("out", optical_out.shape)
            refine_out = unet_model(optical_out.to(device))
            
            out = hp.crop_out_training(refine_out, target_height, target_width, image_shift_height, image_shift_width)
            gt = sample_batched['im_gt'].unsqueeze(0).to(device)
            
            gt = hp.crop_out_training(gt, target_height, target_width, image_shift_height, image_shift_width)
            gt = gt.permute(1, 0, 2, 3)
           # output_numpy = out.detach().cpu().numpy()[0][20]

           # print("output_numpy", output_numpy.shape)
           # print("out", out.shape)
            #print("gt", gt.shape)

            Gloss(output=out, target=gt)
            Gloss.total_loss.backward()
            optical_optimizer.step()
            digital_optimizer.step()
            
            print('epoch: ', itr, ' batch: ', i_batch, 'LR: ', args.digital_lr, args.optical_lr, ' loss: ', Gloss.total_loss.item(), end='\r')
            
        output_numpy = out.detach().cpu().numpy()[0][0]
        gt_numpy = gt.detach().cpu().numpy()[0][0]
        meas_numpy = optical_out.detach().cpu().numpy()[0][0]
        
        '''
        for iii in range(0,N_B):
            output_numpy = out.detach().cpu().numpy()[0][iii]
            gt_numpy = gt.detach().cpu().numpy()[0][iii]
            meas_numpy = optical_out.detach().cpu().numpy()[0][iii]
            im_gt = Image.fromarray((np.clip(gt_numpy/np.max(gt_numpy),0,1)*255).astype(np.uint8))
            im = Image.fromarray((np.clip(output_numpy/np.max(output_numpy),0,1)*255).astype(np.uint8))
            im_meas = Image.fromarray((np.clip(meas_numpy/np.max(meas_numpy),0,1)*255).astype(np.uint8))
            im.save(filepath_save + str(iii) + '.png')
            im_gt.save(filepath_save + str(iii) + 'gt.png')
            im_meas.save(filepath_save + str(iii) + 'meas.png')
            '''
        
        if args.save_checkponts == True:
            h_out = h_out.detach().cpu().numpy()
            a_zernike_out = a_zernike_out.detach().cpu().numpy()
            PSFs_out = PSFs_out.detach().cpu().numpy()
            
            torch.save(optical_model.state_dict(), filepath_save + 'optical_model_noval.pt')
            torch.save(unet_model.state_dict(), filepath_save + 'unet_model_noval.pt')
            np.savetxt(filepath_save + 'HeightMap_noval.txt', h_out)
            np.savetxt(filepath_save + 'a_zernike_noval.txt', a_zernike_out)
            np.save(filepath_save + 'PSFs_noval.npy', PSFs_out)
        #    torch.save(D_model.state_dict(), filepath_save + 'D_model_noval.pt')
        
        
        if itr%1==0:
            total_loss=0
            for i_batch, sample_batched in enumerate(dataloader_test):
                with torch.no_grad():
                    [optical_out,a_zernike_out, h_out, PSFs_out] = optical_model(sample_batched['im_gt'].to(device))
                    refine_out = unet_model(optical_out.to(device))
                    out = hp.crop_out_training(refine_out, target_height, target_width, image_shift_height, image_shift_width)
                    gt = sample_batched['im_gt'].unsqueeze(0).to(device)
                    gt = hp.crop_out_training(gt, target_height, target_width, image_shift_height, image_shift_width)
                    gt = gt.permute(1, 0, 2, 3)

                    Gloss(output=out, target=gt)
                    total_loss+=Gloss.total_loss.item()
                    
                  #  print('loss for testing image ',itr,' ',i_batch, Gloss.total_loss.item())

            print('Total loss for testing set ',itr,' ', total_loss)
                 
                 
            if args.save_checkponts == True:
                im_gt = Image.fromarray((np.clip(gt_numpy/np.max(gt_numpy),0,1)*255).astype(np.uint8))
                im = Image.fromarray((np.clip(output_numpy/np.max(output_numpy),0,1)*255).astype(np.uint8))
                im_meas = Image.fromarray((np.clip(meas_numpy/np.max(meas_numpy),0,1)*255).astype(np.uint8))
                im.save(filepath_save + str(itr) + '.png')
                im_gt.save(filepath_save + str(itr) + 'gt.png')
                im_meas.save(filepath_save + str(itr) + 'meas.png')
                SL.save_loss_to_file(itr, total_loss, filename = filepath_save + 'loss_history.txt')
            
            
            if total_loss<best_loss:
                best_loss=total_loss
    
                # save checkpoint
                if args.save_checkponts == True:
                    torch.save(optical_model.state_dict(), filepath_save + 'optical_model.pt')
                    torch.save(unet_model.state_dict(), filepath_save + 'unet_model.pt')
                    
                    h_out = h_out.detach().cpu().numpy()
                    a_zernike_out = a_zernike_out.detach().cpu().numpy()
                    PSFs_out = PSFs_out.detach().cpu().numpy()
                    np.savetxt(filepath_save + 'HeightMap.txt', h_out)
                    np.savetxt(filepath_save + 'a_zernike.txt', a_zernike_out)
                    np.save(filepath_save + 'PSFs.npy', PSFs_out)
                    np.savetxt(filepath_save + 'best_loss.txt', [best_loss])
                   # torch.save(D_model.state_dict(), filepath_save + 'D_model.pt')
                    
        stop = time.time()
        print('Total time for this epoch: ',stop-start,'s')
        
