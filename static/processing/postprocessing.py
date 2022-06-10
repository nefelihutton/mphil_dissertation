import os
import numpy as np

from scipy import ndimage as ndi

def postprocessing(img):
    """
    Input is a binary image (from output of inference) - so you have to change from COCO to binary
    1. images as an np array
    2. makes a distance matrix 
    3. will filter the distance matrix
    Output is a filtered distance matrix binary which is a skinnier version
    """
    lc_img_array = np.asarray(img)
    
    distance_matrix = ndi.distance_transform_edt(lc_img_array)

    for i in range(len(distance_matrix)):
        ind_low = np.where(distance_matrix[i] < 5)
        distance_matrix[i][ind_low] = 0
        ind_high = np.where(distance_matrix[i] >= 5)
        distance_matrix[i][ind_high] = 1
    
    return -distance_matrix
