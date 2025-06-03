import torch.nn as nn
import models.loss_L1L2 as L

class MyEnsemble(nn.Module):
    def __init__(self, unet_model):
        super(MyEnsemble, self).__init__()
        self.unet_model = unet_model
    def forward(self, x):
        #fft_numpy = x.detach().cpu().numpy()[0]
        #print(fft_numpy.shape)
        final_output = self.unet_model(x)
        return final_output

class MyEnsembleParallel(nn.Module):
    def __init__(self, unet_model, loss):
        super(MyEnsembleParallel, self).__init__()
        self.unet_model = unet_model
        self.loss = loss

    def forward(self, x, gt):
        output = self.unet_model(x)
        G_loss = self.loss(output, gt)
        #output_loss = G_loss.total_loss

        return G_loss #output_loss
    
class MyEnsembleCombine(nn.Module):
    def __init__(self, channel_processing, unet_model):
        super(MyEnsembleCombine, self).__init__()
        self.channel_processing = channel_processing
        self.unet_model = unet_model

    def forward(self, x):
        channels = self.channel_processing(x)
        output = self.unet_model(channels)

        return output #output_loss