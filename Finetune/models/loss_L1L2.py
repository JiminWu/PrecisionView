import torch
from torch import nn
import torch.nn.functional as F
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

def compute_gradients(image):
    """
    Compute the gradients of an image using Sobel filters.
    
    Parameters:
    image (torch.Tensor): Input image tensor of shape (B, C, H, W).
    
    Returns:
    torch.Tensor: Gradient in the x direction.
    torch.Tensor: Gradient in the y direction.
    """
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    sobel_y = torch.tensor([[-1, -2, -1],
                            [0,  0,  0],
                            [1,  2,  1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    sobel_x = sobel_x.to(image.device)
    sobel_y = sobel_y.to(image.device)
    
    sobel_x = sobel_x.repeat(image.size(1), 1, 1, 1)
    sobel_y = sobel_y.repeat(image.size(1), 1, 1, 1)
    
    grad_x = F.conv2d(image, sobel_x, padding=1, groups=image.size(1))
    grad_y = F.conv2d(image, sobel_y, padding=1, groups=image.size(1))
    
    return grad_x, grad_y

def gradient_L1_loss(output, target):

    pred_grad_x, pred_grad_y = compute_gradients(output)
    target_grad_x, target_grad_y = compute_gradients(target)
    
    grad_diff_x = pred_grad_x - target_grad_x
    grad_diff_y = pred_grad_y - target_grad_y
    
    loss_x = torch.mean(torch.abs(grad_diff_x))
    loss_y = torch.mean(torch.abs(grad_diff_y))
    
    gradient_loss = loss_x + loss_y
    
    return gradient_loss

def gradient_L2_loss(output, target):

    pred_grad_x, pred_grad_y = compute_gradients(output)
    target_grad_x, target_grad_y = compute_gradients(target)
    
    grad_diff_x = pred_grad_x - target_grad_x
    grad_diff_y = pred_grad_y - target_grad_y
    
    loss_x = torch.mean(grad_diff_x ** 2)
    loss_y = torch.mean(grad_diff_y ** 2)
    
    gradient_loss = loss_x + loss_y
    
    return gradient_loss

class GLoss(nn.Module):
    def __init__(self, args):
        super(GLoss, self).__init__()
        self.args = args

    def forward(self, output: "Tensor", target: "Tensor"):
        device = output.device

        self.total_loss = torch.tensor(0.0).to(device)
        self.L1_loss = torch.tensor(0.0).to(device)
        self.L2_loss = torch.tensor(0.0).to(device)
        self.DL1_loss = torch.tensor(0.0).to(device)
        self.DL2_loss = torch.tensor(0.0).to(device)
        self.SSIM_loss = torch.tensor(0.0).to(device)
        self.RMS_loss = torch.tensor(0.0).to(device)
        
        if self.args.lambda_L1:
            l1_loss = torch.nn.L1Loss()#.to(rank)
            self.L1_loss += (l1_loss(output, target) * self.args.lambda_L1)
            
        if self.args.lambda_L2:
            l2_loss = torch.nn.MSELoss()#.to(rank)
            self.L2_loss += (l2_loss(output, target) * self.args.lambda_L2)
        
        if self.args.lambda_DL1:
            self.DL1_loss += (gradient_L1_loss(output, target) * self.args.lambda_DL1)
            
        if self.args.lambda_DL2:
            self.DL2_loss += (gradient_L2_loss(output, target) * self.args.lambda_DL2)
            
        if self.args.lambda_RMS:
            self.RMS_loss += F.mse_loss(output, target).mean() * self.args.lambda_RMS
        
        if self.args.lambda_SSIM:
            self.SSIM_loss += ((1 - ms_ssim(output, target)) * self.args.lambda_SSIM)
            #self.SSIM_loss = SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=1)


        self.total_loss += (
           # self.adversarial_loss
            self.L1_loss
            + self.L2_loss
            + self.DL1_loss
            + self.DL2_loss
            + self.SSIM_loss
            + self.RMS_loss
        )

        return self.total_loss
