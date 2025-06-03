import numpy as np

def save_loss_to_file(epoch_num, loss, filename='loss_history.txt'):
    with open(filename, 'a') as file:
        file.write('epoch #'+ str(epoch_num) + ', loss ' + str(loss) + '\n')

def save_param_to_file(image_num, PSNR, SSIM, filename='param_history.txt'):
    with open(filename, 'a') as file:
        file.write('Image #'+ str(image_num) + ', PSNR ' + str(PSNR) + ', SSIM ' + str(SSIM) + '\n')
        
def save_avgparam_to_file(PSNR_avg, PSNR_std, SSIM_avg, SSIM_std, filename='avgparam_history.txt'):
    with open(filename, 'a') as file:
        file.write('Average PSNR '+ str(PSNR_avg) + ', STD ' + str(PSNR_std) + ', Average SSIM ' + str(SSIM_avg) + ', STD ' + str(SSIM_std) + '\n')
