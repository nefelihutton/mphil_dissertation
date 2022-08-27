
import os 
import numpy as np
import matplotlib.pyplot as plt
import math
from math import degrees, atan2

from numpy import array
from numpy import linalg as la

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
  

def find_eccentricity(points_2d):
  """
  Input: 2d coordinates (points_2d)
  Output: e (cell eccentricity)
  """
  X = np.stack((points_2d[:,0], points_2d[:,1]), axis=0)
  covariance_matrix = np.cov(X)
  eigen_vals, eigen_vecs = np.linalg.eig(covariance_matrix)
  if eigen_vals[0] > eigen_vals[1]:
    major = eigen_vals[0]
    minor = eigen_vals[1]
  else:
      major = eigen_vals[1]
      minor = eigen_vals[0]
  e = minor/major
  
  return e

def find_orientation(points_2d):
  """
  Input: 2d coordinates (points_2d)
  Output: degrees (cell orientation), e (cell eccentricity)
  """
  X = np.stack((points_2d[:,0], points_2d[:,1]), axis=0)
  covariance_matrix = np.cov(X)
  eigen_vals, eigen_vecs = np.linalg.eig(covariance_matrix)

  idx = np.argmax(eigen_vals)
  y_axis = [0, 1]
  vector_2 = eigen_vecs[:,idx]

  unit_vector_1 = y_axis / np.linalg.norm(y_axis)
  unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
  dot_product = np.dot(unit_vector_1, unit_vector_2)
  angle = np.arccos(dot_product)

  degrees = math.degrees(angle)
  # if degrees > 90: 
  #   degrees = degrees - 180
  
  e = eigen_vals[0]/eigen_vals[1]

  return degrees, e

