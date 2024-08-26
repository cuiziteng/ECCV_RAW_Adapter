import imghdr
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np

def value_encode(value, max, min):
    return (value-min)/(max-min)

def value_decode(value, max ,min):
    return value*(max-min)+min

def pixel_project_2d(img, normal):
    # Step 1: Project on Surface
    img = img.permute(0,2,3,1) # (B, H, W, C)
    R, G, B = img[:, :, :, 0], img[:, :, :, 1], img[:, :, :, 2]
    # Assume Luminance Line Pass by the Original Point (0, 0, 0)
    d = 0
    square = normal[0]**2 + normal[1]**2 + normal[2]**2
    t = (normal[0]*R + normal[1]*G + normal[2]*B + d)/square
    
    # Min and Max of parameter t
    t_max = (normal[0] + normal[1] + normal[2] + d)/square
    t_min = d/square
    
    R_new = R - normal[0]*t 
    G_new = G - normal[1]*t
    B_new = B - normal[2]*t
    
    t = value_encode(t, t_max, t_min)
    # Optional: Gamma Encoding
    
    img_lut = torch.stack([R_new, G_new, B_new], dim=-1)
    return [t, t_max, t_min], img_lut


def pixel_project_back(t_s, img_lut, normal):
    
    # decode t back to the original range
    t = value_decode(t_s[0], t_s[1], t_s[2])

    R_new, G_new, B_new = img_lut[:, :, :, 0], img_lut[:, :, :, 1], img_lut[:, :, :, 2]
    R_re, G_re, B_re = R_new+normal[0]*t, G_new+normal[1]*t, B_new+normal[2]*t
    img_out = torch.stack([R_re, G_re, B_re], dim=-1)  # (B, H, W, C)
    return img_out.permute(0,3,1,2)

def LUT_mapping_1d(t, lut_1d):
    B, H, W = t.shape
    N = lut_1d.shape[0] 
    t = (t * N).to(torch.int32)
    
    lut_1d = torch.cumsum(lut_1d, dim=0)
    lut_1d = ((lut_1d-torch.min(lut_1d)) / (torch.max(lut_1d)-torch.min(lut_1d)))* N
    lut_1d = lut_1d.to(torch.int32)
    
    N_new = H * W
    id_t = t.to(torch.long).view(B, N_new)
    out = lut_1d.reshape(N)[id_t].reshape(B, H, W)
    
    return out 

# Unprocess RGB to RAW
# camera color matrix
xyz2cams = [[[1.0234, -0.2969, -0.2266],
                [-0.5625, 1.6328, -0.0469],
                [-0.0703, 0.2188, 0.6406]],
            [[0.4913, -0.0541, -0.0202],
                [-0.613, 1.3513, 0.2906],
                [-0.1564, 0.2151, 0.7183]],
            [[0.838, -0.263, -0.0639],
                [-0.2887, 1.0725, 0.2496],
                [-0.0627, 0.1427, 0.5438]],
            [[0.6596, -0.2079, -0.0562],
                [-0.4782, 1.3016, 0.1933],
                [-0.097, 0.1581, 0.5181]]]
rgb2xyz = [[0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]]

def apply_ccm(image, ccm):
    '''
    The function of apply CCM matrix
    '''
    shape = image.shape
    image = image.view(image.shape[0], -1, 3)
    image = torch.tensordot(image, ccm, dims=[[-1], [-1]])
    return image.view(shape)


def mosaic(img):
    """Extracts RGGB Bayer planes from an RGB image."""
    #image.shape.assert_is_compatible_with((None, None, 3))
    #shape = tf.shape(image)
    shape = img.shape
    
    red = img[:, 0::2, 0::2, 0]
    green_red = img[:, 0::2, 1::2, 1]
    green_blue = img[:, 1::2, 0::2, 1]
    blue = img[:, 1::2, 1::2, 2]
    
    raw = torch.stack([red, green_red, green_blue, blue], dim=-1)
    raw = torch.reshape(raw, (img.shape[0], img.shape[1]//2, img.shape[2]//2, 4))
    
    return raw

# 1.inverse tone, 2.inverse gamma, 3.sRGB2cRGB, 4.inverse WB digital gains
def Unprocess(img):
    
    img1 = img.permute(0,2,3,1) # (B, H, W, C)
    # inverse tone mapping
    img1 = 0.5 - torch.sin(torch.asin(1.0 - 2.0 * img1) / 3.0)
    
    # inverse gamma
    epsilon = torch.FloatTensor([1e-8]).to(img.device)
    gamma = random.uniform(2.0, 3.5)
    img2 = torch.max(img1, epsilon) ** gamma
    
    # sRGB2cRGB
    xyz2cam = random.choice(xyz2cams)
    rgb2cam = np.matmul(xyz2cam, rgb2xyz)
    rgb2cam = torch.from_numpy(rgb2cam / np.sum(rgb2cam, axis=-1)).to(torch.float).to(img.device)
    img3 = apply_ccm(img2, rgb2cam)
    
    # Mosaicing
    img4 = mosaic(img3).permute(0,3,1,2)
    
    return img4

if __name__ == '__main__':
    img_input = torch.rand([2,3,400,600])
    img_out = Unprocess(img_input)
    print(img_out.shape)



