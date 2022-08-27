# texture i.e. standard deviation of intensity (STD)

# 1. need to obtain intensity at every pixel in the mask 

from edges import find_coords
import matplotlib.pyplot as plt
import numpy as np


def texture(img_fname, seg_file):
    """
    Extract pixel values from image (2d np array) at specific 
    coordinates mask coordinates (2d np array)
    img_fname = "path/to/image"
    seg_file = "path/to/seg.py"
    """
    
    img = plt.imread(img_fname)
    np_img = np.asarray(img)

    seg_mask = np.load(seg_file)
    mask_loc = find_coords(seg_mask[0])
    pixel_values = []
    for i in range(len(mask_loc)):
        pix = mask_loc[i]
        pix_val = np_img[pix[0], pix[1]][0]
        pixel_values.append(pix_val)
    pix_std = np.std(pixel_values)
    
    return pix_std

