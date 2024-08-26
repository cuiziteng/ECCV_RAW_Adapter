import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
import PIL.Image as Image
from tqdm import tqdm
import random
import rawpy
import torchvision

raw_path = r'/data/unagi0/cui_data/light_dataset/PASCAL_RAW/original/raw'

out_path = r'/data/unagi0/cui_data/light_dataset/PASCAL_RAW/original/demosaic'
out_path_low = r'/data/unagi0/cui_data/light_dataset/PASCAL_RAW/PASCALRAW/demosaic_low'
out_path_oe = r'/data/unagi0/cui_data/light_dataset/PASCAL_RAW/PASCALRAW/demosaic_oe'

def mkdir(path):
    if not os.path.join(path):
        os.makedirs(path)

def random_noise_levels():
    """Generates random shot and read noise from a log-log linear distribution."""
    log_min_shot_noise = np.log(0.0001)
    log_max_shot_noise = np.log(0.012)
    log_shot_noise = np.random.uniform(log_min_shot_noise, log_max_shot_noise)
    shot_noise = np.exp(log_shot_noise)

    line = lambda x: 2.18 * x + 1.20
    log_read_noise = line(log_shot_noise) + np.random.normal(scale=0.26)
    # print('shot noise and read noise:', log_shot_noise, log_read_noise)
    read_noise = np.exp(log_read_noise)
    return shot_noise, read_noise

def low_light_trans(raw):
    lower, upper = 0.05, 0.4    # low-light range
    exposure_value = random.uniform(lower, upper) 
    raw_low_light = raw * exposure_value

    shot_noise, read_noise = random_noise_levels()
    var = raw_low_light * shot_noise + read_noise
    var = torch.max(var, torch.FloatTensor([1e-5]))
    noise = torch.normal(mean=0, std=torch.sqrt(var))
    raw_low_light = raw_low_light + noise
    return raw_low_light

def oe_light_trans(raw):
    lower, upper = 3.5, 5.0     # over-exp range
    exposure_value = random.uniform(lower, upper) 
    raw_over_exp = raw * exposure_value

    shot_noise, read_noise = random_noise_levels()
    var = raw_over_exp * shot_noise + read_noise
    var = torch.max(var, torch.FloatTensor([1e-5]))
    noise = torch.normal(mean=0, std=torch.sqrt(var))
    raw_over_exp = raw_over_exp + noise
    return raw_over_exp

os.makedirs(out_path, exist_ok=True)
os.makedirs(out_path_low, exist_ok=True)
os.makedirs(out_path_oe, exist_ok=True)


if __name__ == '__main__':

    for file in tqdm(os.listdir(raw_path)):
        
        image_name = os.path.join(raw_path, file)

        # Setp 0: Load RAW data
        raw = rawpy.imread(image_name)
        mosaic = raw.raw_image
        black = mosaic.min()
        saturation = mosaic.max()

        # Step 1: Linearization
        uint12_max = 2**12 - 1
        mosaic -= black  # black subtraction
        mosaic *= int(uint12_max/(saturation - black))
        mosaic = np.clip(mosaic, 0, uint12_max)  # clip to range
        
        mosaic = np.float64(mosaic) 

        # Step 2: Demosacing, RGGB --> RGB
        def demosaic(m):    
            r = m[0::2, 0::2]
            g = np.clip(m[0::2, 1::2]//2 + m[1::2, 0::2]//2,
                        0, 2 ** 12 - 1)
            b = m[1::2, 1::2]
            return np.dstack([r, g, b])

        raw_rgb = demosaic(mosaic) 

        # Step 3: Resize to match PASCAL RAW Label, 600*400 resolution
        raw_rgb_resize = cv2.resize(raw_rgb, (600, 400),interpolation=cv2.INTER_CUBIC)  # match detection labels in PASCAL
        raw_rgb_resize = np.clip(raw_rgb_resize, 0, 2**12-1)

        raw_rgb_resize = raw_rgb_resize/(2**12-1)

        ## Normal-Light Scene 
        plt.imsave(os.path.join(out_path, file.replace('.nef', '.png')), np.float64(np.clip(raw_rgb_resize, 0, 1)))
        raw_rgb_resize_t = torch.from_numpy(raw_rgb_resize).float().permute(2,0,1).unsqueeze(0)
        
        ## Low-Light Scene 
        raw_rgb_low = low_light_trans(raw_rgb_resize_t)
        torchvision.utils.save_image(raw_rgb_low, os.path.join(out_path_low, file.replace('.nef', '.png')))

        ## Over-Exposure Scene 
        raw_rgb_oe = oe_light_trans(raw_rgb_resize_t)
        torchvision.utils.save_image(raw_rgb_oe, os.path.join(out_path_oe, file.replace('.nef', '.png')))
        

