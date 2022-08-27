# compactness, solidity

import math
from edges import find_coords
from scipy.spatial import ConvexHull
import numpy as np

def compactness(perimeter_arr, area_arr):
  """
  compactness takes in 
  perimeter_arr and area_arr
  """
  comp_arr = [(perimeter^2)/(4*math.pi*area) for area in area_arr for perimeter in perimeter_arr] 
  return comp_arr

def solidity(seg_mask, area_arr):
  """
  Find solidity of a shape
  """
  mask_loc = [find_coords(mask) for mask in seg_mask]
  hull_area = [ConvexHull(point_cloud).area for point_cloud in mask_loc]
  solidity_arr = [np.abs(hull - area) for area in area_arr for hull in hull_area]
  return solidity_arr

