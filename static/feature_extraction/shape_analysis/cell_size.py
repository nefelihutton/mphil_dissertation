import numpy as np
import alphashape

import cv2
import matplotlib.pyplot as plt
import shapely
import shapely.geometry
from shapely.geometry import Point

from edges import flatten

# Cell size defined by mask area, perimeter, Feret's diameter
# Faster implementation of below 
# read in seg.npy and then make a list of contours for each time frame

def get_contours(one_frame_seg):
  contours_list = []
  for mask in one_frame_seg:
    image = mask*255
    image = image.astype('uint8')
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_list.append(contours)
  return contours_list

def get_mask_area(contours_list):
  """
  Input: shape array 
  returns pixel squared area of the inferred masks
  """
  area_list = [cv2.contourArea(cv2_contours[0]) for cv2_contours in contours_list]
  return area_list

def get_mask_perimeter(contours_list):
  """
  Input: shape array 
  returns pixel squared area of the inferred masks
  """
  peri_list = [cv2.arcLength(cv2_contours[0], True) for cv2_contours in contours_list]
  return peri_list

 
# Solved using convex hull algorithm implemented with alphashpe (slow)
def find_alphashape(seg_mask):
  """
  input - segmentation mask output (numpy array) i.e. predicted output on images
  output - shapely shape array of all masks
  """
  shape_arr = []
  for i in range(len(seg_mask)):
    states = seg_mask[i]
    A_true = np.where(states)
    points_2d = np.vstack((A_true[0],A_true[1])).T

    alpha_shape = alphashape.alphashape(points_2d, 0.5)
  
    if alpha_shape.type == 'Polygon':
      shape = alpha_shape
    else:
      shape = np.array([x for x in alpha_shape.geoms])
    shape_arr.append(shape)
  shape_arr = np.hstack(shape_arr)
  return shape_arr

def find_mask_area(shape_arr):
  """
  Input: shape array 
  returns pixel squared area of the inferred masks
  """
  area_arr = [x.area for x in shape_arr]
  return area_arr

def find_mask_perimeter(shape_arr):
  """
  Input: shape array 
  returns pixel length/perimeter of the inferred masks
  """
  perimeter_arr = [x.length for x in shape_arr]
  return perimeter_arr

def feret_diameter(shape_arr):
  box_arr = [polygon.minimum_rotated_rectangle for polygon in shape_arr]
  ratio = []
  for box in box_arr: 
    x, y = box.exterior.coords.xy
    edge_length = (Point(x[0], y[0]).distance(Point(x[1], y[1])), Point(x[1], y[1]).distance(Point(x[2], y[2])))
    l = max(edge_length)
    w = min(edge_length)
    r = l/w
    ratio.append(r)
  return ratio

