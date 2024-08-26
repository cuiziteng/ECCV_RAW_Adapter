# from .block import Matrix_Predictor, NILUT
from genericpath import exists
import torch.nn as nn
import torch
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torchvision

from .block import Matrix_Predictor, Kernel_Predictor, NILUT
from .kernel import gaussian_blur

# Shades of Gray and Colour Constancy (Graham D. Finlayson, Elisabetta Trezzi)
def SoG_algo(img, p=1):
    # https://library.imaging.org/admin/apis/public/api/ist/website/downloadArticle/cic/12/1/art00008
    img = img.permute(1,2,0)       # (C,H,W) --> (H,W,C)

    img_P = torch.pow(img, p)
    
    R_avg = torch.mean(img_P[:,:,0]) ** (1/p)
    G_avg = torch.mean(img_P[:,:,1]) ** (1/p)
    B_avg = torch.mean(img_P[:,:,2]) ** (1/p)

    Avg = torch.mean(img_P) ** (1/p)

    R_avg = R_avg / Avg
    G_avg = G_avg / Avg
    B_avg = B_avg / Avg

    img_out = torch.stack([img[:,:,0]/R_avg, img[:,:,1]/G_avg, img[:,:,2]/B_avg], dim=-1)

    return img_out 

def Gain_Denoise(I1, r1, r2, gain, sigma, k_size=3):  # [9, 9] in LOD dataset, [3, 3] in other dataset
    out = []
    for i in range(I1.shape[0]):
        I1_gain = gain[i] * I1[i,:,:,:]
        blur = gaussian_blur(I1_gain, \
                                [k_size, k_size], \
                                [r1[i], r2[i]])
        sharp = blur + sigma[i] * (I1[i,:,:,:] - blur)
        out.append(sharp)
    return torch.stack([out[i] for i in range(I1.shape[0])], dim=0)


def WB_CCM(I2, ccm_matrix, distance):
    out_I3 = []
    out_I4 = []
    for i in range(I2.shape[0]):
        # SOG White Balance Algorithm
        I3 = SoG_algo(I2[i,:,:,:], distance[i])
        
        # Camera Color Matrix
        I4 = torch.tensordot(I3, ccm_matrix[i,:,:], dims=[[-1], [-1]])
        I4 = torch.clamp(I4, 1e-5, 1.0)
         
        out_I3.append(I3)
        out_I4.append(I4)

    return  torch.stack([out_I3[i] for i in range(I2.shape[0])], dim=0), \
            torch.stack([out_I4[i] for i in range(I2.shape[0])], dim=0)

def check(input_tensor, save_path):
    os.makedirs(save_path, exist_ok=True)
    for i in range(input_tensor.shape[0]):
        torchvision.utils.save_image(input_tensor[i,:,:,:], os.path.join(save_path, str(i)+'.png'))

# Input-level Adapter
class Input_level_Adapeter(nn.Module):
    def __init__(self, mode='normal', lut_dim=32, out='all', k_size=3, w_lut=True): 
        super(Input_level_Adapeter, self).__init__()
        '''
        mode: normal (for normal & over-exposure conditions) or low (for low-light conditions)
        lut_dim: implicit neural look-up table dim number
        out: if all, return I1, I2, I3, I4, I5, if not all, only return I5
        k_size: denosing kernel size, must be odd number, we set it to 9 in LOD dataset and 3 in other dataset
        w_lut: with or without implicit 3D Look-up Table
        '''

        self.Predictor_K = Kernel_Predictor(dim=64, mode=mode)
        self.Predictor_M = Matrix_Predictor(dim=64)
        self.w_lut = w_lut
        if self.w_lut:
            self.LUT = NILUT(hidden_features=lut_dim)    
        self.out = out
        self.k_size = k_size
        


    def forward(self, I1):
        # (1). I1 --> I2: Denoise & Enhancement & Sharpen
        r1, r2, gain, sigma = self.Predictor_K(I1)
        I2 = Gain_Denoise(I1, r1, r2, gain, sigma, k_size=self.k_size)  # (B,C,H,W)
        I2 = torch.clamp(I2, 1e-5, 1.0) # normal & over-exposure
        
        ccm_matrix, distance = self.Predictor_M(I2)
        # (2). I2 --> I3: White Balance, Shade of Gray
        # (3). I3 --> I4: Camera Colour Matrix Transformation
        I3, I4 = WB_CCM(I2, ccm_matrix, distance) # (B,H,W,C)
        
        if self.w_lut:
        # (4). I4 --> I5: Implicit Neural LUT
            I5 = self.LUT(I4).permute(0,3,1,2)
            
            if self.out == 'all':   # return all features
                return [I1, I2, I3.permute(0,3,1,2), I4.permute(0,3,1,2), I5]
            else:   # only return I5
                return [I5]
        
        else:
            if self.out == 'all':
                return [I1, I2, I3.permute(0,3,1,2), I4.permute(0,3,1,2)]
            else:
                return [I4.permute(0,3,1,2)]


        
        
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']='4'

    input = torch.rand([4,3,512,512])
    net = Input_level_Adapeter(out='all', w_lut=False)
    out = net(input)

    