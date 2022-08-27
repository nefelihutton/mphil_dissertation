import os 
from scipy.signal import find_peaks
import numpy as np

import sys
import pandas as pd

from descartes import PolygonPatch
import matplotlib.pyplot as plt

import alphashape

import math
from math import degrees, atan2

def get_boundary_coords(contours_list):
  """
  Input: shape array 
  returns pixel squared area of the inferred masks
  """
  boundary_coords_list = [np.vstack((cv2_contours[0][:,0][:,0], cv2_contours[0][:,0][:,1])).T for cv2_contours in contours_list]
  return boundary_coords_list


def find_coords(mask):
  """
  Input: seg_mask output from inference
  Output: x y coordinates of the mask (x,y)
  """
  A_true = np.where(mask)
  x = []
  y = []
  for idx in range(0, A_true[0].shape[0]):
    x.append(A_true[0][idx])
    y.append(A_true[1][idx])
  
  points_2d = np.vstack((x,y)).T

  return points_2d

def coord_to_edge(points_2d):
  alpha_shape = alphashape.alphashape(points_2d, 0.5)
  if alpha_shape.type == 'Polygon':
      edge = np.array(alpha_shape.exterior.coords)
  else:
    mask_edge = np.array([list(x.exterior.coords) for x in alpha_shape.geoms])

  return mask_edge

def mask_to_edges(seg_mask):
  """
  Input: segm_mask output from inference
  returns 2d edge coordinates of the inferred masks
  """
  mask_edges = []
  for i in range(len(seg_mask)):
    print(i)
    states = seg_mask[i]
    A_true= np.where(states)

    x = []
    y = []

    for idx in range(0, A_true[0].shape[0]):
      x.append(A_true[0][idx])
      y.append(A_true[1][idx])

    points_2d = np.vstack((x,y)).T
    alpha_shape = alphashape.alphashape(points_2d, 0.5)

    if alpha_shape.type == 'Polygon':
      edge = np.array(alpha_shape.exterior.coords)
    else:
      edge = np.array([list(x.exterior.coords) for x in alpha_shape.geoms])
      
    mask_edges.append(edge)

  return mask_edges

def find_edge(shape_arr):
  mask_edge = [[x.exterior.coords] for x in shape_arr]
  return mask_edge

def clockwise_angle(x, y, center_x, center_y):
    """
	Input: 
	x: x coordinates of edge
	y: y coordinates of edge
	Center_x: average point of x 
	center_y: average point of y

	Returns: Bearing from centroid to outer edge point  
	"""
    angle = degrees(atan2(y - center_y, x - center_x))
    bearing = (90 - angle) % 360
    return bearing

def get_distance_centroid(edge_array):
  euc_distance = []
  angles = []
  centroid_x = np.average(edge_array[:,0])
  centroid_y = np.average(edge_array[:,1])

  for i in range(0, len(edge_array)):
    x = edge_array[:,0][i]
    y = edge_array[:,1][i]
    distance = math.sqrt(math.pow((x - centroid_x), 2) + math.pow((y - centroid_y), 2))

    point_ = (x, y)
    centroid_ = (centroid_x, centroid_y)
    angle_find = clockwise_angle(x, y, centroid_x, centroid_y)
    euc_distance.append(distance)
    angles.append(angle_find)
    shape_signature = np.vstack((angles,euc_distance)).T
  return shape_signature

from collections import Iterable
def flatten(lis):
     for item in lis:
         if isinstance(item, Iterable) and not isinstance(item, str):
             for x in flatten(item):
                 yield x
         else:        
             yield item