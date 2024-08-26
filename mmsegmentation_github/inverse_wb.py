# Inverse White Balance Process 
# from ICCV 2021: Multitask AET with Orthogonal Tangent Regularity for Dark Object Detection

import numpy as np
import os
import torch
import torch.nn as nn
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--input_dir", default='', type=str, help="")
parser.add_argument("--output_dir", default='', type=str, help="")
config = parser.parse_args()

if not os.path.exists(config.output_dir):
    os.makedirs(config.output_dir)

rgb_range = [0.8, 0.1]
red_range = [1.9, 2.4]
blue_range = [1.5, 1.9]

for file in tqdm(os.listdir(config.input_dir)):
    img_name = os.path.join(config.input_dir, file)
    img = plt.imread(img_name)/255.0
    
    red_gain = random.uniform(red_range[0], red_range[1])
    blue_gain = random.uniform(blue_range[0], blue_range[1])

    gains1 = np.stack([1.0 / red_gain, 1.0, 1.0 / blue_gain])
    
    gains1 = gains1[np.newaxis, np.newaxis, :]
    
    img_out = img * gains1
    
    img_out = np.clip(img_out, 0, 1)
    
    plt.imsave(os.path.join(config.output_dir, file), img_out)
    
    
