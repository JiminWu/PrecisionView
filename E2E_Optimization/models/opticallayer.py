import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

#########  generate out-of-focus phase  ###############
## @param{Phi_list}: a list of Phi values
## @param{N_B}: size of the blur kernel
## @return{OOFphase}
def gen_OOFphase(Phi_list, N_B):
    # return (Phi_list,pixel,pixel,color)
    N = N_B
    x0 = np.linspace(-1.09, 1.09, N) # 25/23 =1.09
    xx, yy = torch.tensor(np.array(np.meshgrid(x0, x0))).to(device="cuda:0")
    OOFphase = torch.tensor(np.empty([len(Phi_list), N, N, 1], dtype=np.float32)).to(device="cuda:0")
    for j in range(len(Phi_list)):
        Phi = Phi_list[j]
        OOFphase[j, :, :, 0] = Phi * (xx ** 2 + yy ** 2)
    return OOFphase



##################  Generates the PSFs  ########################
## @param{h}: height map of the mask
## @param{OOFphase}: out-of-focus phase
## @param{wvls}: wavelength \lambda
## @param{idx}: index of the PSF
## @param{N_B}: size of the blur kernel
#################################################################
def gen_PSFs(h, OOFphase, wvls, idx, N_B):
    n = 1.5  # diffractive index

    OOFphase_B = OOFphase[:, :, :, 0]
    phase_B = 2 * np.pi / wvls * (n - 1) * h + OOFphase_B
    complex_idx = torch.complex(idx,torch.zeros_like(idx))
    complex_exp = torch.complex(torch.zeros_like(phase_B),phase_B)
    Pupil_B = complex_idx * torch.exp(complex_exp)
 
    Norm_B =  torch.tensor(float(N_B * N_B * torch.sum(idx ** 2)), dtype=torch.float32)
    PSF_B = torch.divide(torch.square(torch.abs(torch.fft.fftshift(torch.fft.fft2(Pupil_B, dim=(-2, -1))))), Norm_B)

    return torch.unsqueeze(PSF_B, -1)

def one_wvl_blur(im, PSFs0):
    im = torch.squeeze(im)
    N_B = PSFs0.shape[1]
    N_Phi = PSFs0.shape[0]
    N_im = im.shape[1]

    padding_sizes_ims = (int(N_B/2), int(N_B/2), int(N_B/2), int(N_B/2))
    sharp = F.pad(im, padding_sizes_ims)

    #print(sharp.shape)
    PSFs = PSFs0.to(torch.float)
    #print(PSFs.shape)
    blurAll = torch.zeros(sharp.size(0), N_im, N_im).to(device="cuda:0")
    
    std_dev = 0.01

    for i in range(sharp.size(0)):
        sharp_single = sharp[i,:,:].unsqueeze(0).unsqueeze(0)
        PSF_single = PSFs[i,:,:].unsqueeze(0).unsqueeze(0)
        blurAll[i, :, :] = F.conv2d(sharp_single, PSF_single, padding=0)
        
        noise = torch.randn_like(blurAll[i, :, :]) * std_dev
        blurAll[i, :, :] = blurAll[i, :, :] + noise
  

  #  print(blurAll.shape)
    blurStack = blurAll.unsqueeze(0)
  #  print(blurAll.shape)

    return blurStack


def blurImage_diffPatch_diffDepth(RGB, PSFs):
    #print(PSFs.shape)
    blur = one_wvl_blur(RGB, PSFs[:, :, :, 0])
    return blur

def add_gaussian_noise(images, std):
    noise = torch.randn_like(images) * std
    return F.relu(images + noise)


class fft_conv(nn.Module):
    
    def __init__(self, gamma, a_zernike_init, Phi_list, N_B, wvls, a_zernike_fix, idx, u2):
        super(fft_conv, self).__init__()
        #psfs = torch.tensor(psfs, dtype=torch.float32)
        gamma = torch.tensor(gamma, dtype=torch.float32)
        u2 = torch.tensor(u2, dtype=torch.float32)
        a_zernike_learn =  a_zernike_init.clone().detach().to(device="cuda:0")
        a_zernike_learn.data = torch.clamp(a_zernike_learn.data, torch.tensor(-wvls / 2).to(device="cuda:0"), torch.tensor(wvls / 2).to(device="cuda:0"))
        #a_zernike_learn = torch.tensor(a_zernike_learn, dtype=torch.float32)
        a_zernike_learn = a_zernike_learn.to(dtype=torch.float32)
       # print(a_zernike_learn.dtype)

        self.Phi_list = torch.tensor(Phi_list).to(device="cuda:0")
        self.N_B = torch.tensor(N_B).to(device="cuda:0")
        self.wvls = torch.tensor(wvls).to(device="cuda:0")
        self.a_zernike_fix = a_zernike_fix.clone().detach().to(device="cuda:0")
        self.idx = torch.tensor(idx).to(device="cuda:0")
       # self.u2 = torch.tensor(u2).to(device="cuda:0")
        
        #self.psfs = nn.Parameter(psfs, requires_grad =True)
        self.gamma = nn.Parameter(gamma, requires_grad =True)
        self.a_zernike_learn = nn.Parameter(a_zernike_learn, requires_grad=True)
        self.u2 =nn.Parameter(u2, requires_grad=True)
        
        
    def forward(self, img):
        
        c = 0 # c: baseline
        OOFphase = gen_OOFphase(self.Phi_list, self.N_B)
        a_zernike = self.a_zernike_learn + self.a_zernike_fix * self.gamma # fixed cubic and learning part
       # print(self.u2.dtype)
       # print(self.a_zernike_learn.dtype)
        g = torch.matmul(self.u2, a_zernike)
        
        h = F.relu(torch.reshape(g, [self.N_B, self.N_B])+c)  # height map of the phase mask, should be all positive
        PSFs = gen_PSFs(h, OOFphase, self.wvls, self.idx, self.N_B)  # return (N_Phi, N_B, N_B, N_color)
        blur = blurImage_diffPatch_diffDepth(img, PSFs)  # size [batch_size * N_Phi, Nx, Ny, 3]
        
        # noise
        sigma = 0.01
        blur_noisy = add_gaussian_noise(blur, sigma)
        blur_noisy = blur_noisy.permute(1, 0, 2, 3)

        return blur_noisy, a_zernike, h, PSFs
    
    
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'u2':self.u2(),
            'a_zernike_learn': self.a_zernike_learn.numpy(),
            'gamma': self.gamma.numpy()
        })
        return config
    

class OpticalEnsemble(nn.Module):
    def __init__(self, opticallayer):
        super(OpticalEnsemble, self).__init__()
        self.opticallayer = opticallayer
        
    def forward(self, x):
        [optical_output,a_zernike_out, h_out, PSFs_out] = self.opticallayer(x)
       # h = h_out.detach().cpu().numpy()[0]
        #        fft_numpy = fft_out.detach().cpu().numpy()[0]
#        print(fft_numpy.shape)
#        fft_numpy = (fft_numpy-fft_numpy.min())/(fft_numpy.max()-fft_numpy.min())
#        for ii in range(5):
#            outimg = fft_numpy[ii,440:1208, 440:1208]
#            outname = [str(ii), '_net.png']
#            plt.imsave("".join(outname), outimg, cmap='gray')
        return optical_output, a_zernike_out, h_out, PSFs_out

