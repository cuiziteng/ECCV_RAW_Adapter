import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", default='', type=str, help="")
parser.add_argument("--output_dir", default='', type=str, help="")
config = parser.parse_args()

# Gray-World White Balance Algorithm
def gray_world(mosaic):
    r = mosaic[:,:,0]
    g = mosaic[:,:,1]
    b = mosaic[:,:,2]
    mu_r = np.average(r)
    mu_g = np.average(g)
    mu_b = np.average(b)
    
    cam_mul = [mu_g/mu_r, 1, mu_g/mu_b]
    
    r *= cam_mul[0]     # scale reds
    b *= cam_mul[2]     # scale blues
    out = np.clip(np.stack([r,g,b],axis=-1), 0, 255.0)# clip to range
    return out


os.makedirs(config.output_dir, exist_ok=True)

for file in tqdm(os.listdir(config.input_dir)):
    img_name = os.path.join(config.input_dir, file)
    img = plt.imread(img_name)/255.0
    img_out = gray_world(img)
    
    plt.imsave(os.path.join(config.output_dir, file), np.clip(img_out,0,1))
