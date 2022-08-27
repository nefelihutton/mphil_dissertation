import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.signal import find_peaks

from scipy.spatial import distance

import os
import matplotlib.pyplot as plt

from helper.timestamp import find_between, find_timestamp


def val_append(dict_obj, key, value):
 if key in dict_obj:
  if not isinstance(dict_obj[key], list):
  # converting key to list type
   dict_obj[key] = [dict_obj[key]]
   # Append the key's value in list
   dict_obj[key].append(value)

# another way of doing the same thing
def globa_id(bbox_dir, iter_bbox):
  cell_id_match = []

  for i in range(len(iter_bbox)-1):
    frame_ids = []
    fir_fr = np.load(os.path.join(bbox_dir, iter_bbox[i]))
    fir_fr_centroid_y = [np.average((fir_fr[n, 0], fir_fr[n, 2])) for n in range(len(fir_fr))]
    fir_fr_centroid_x = [np.average((fir_fr[n, 1], fir_fr[n, 3])) for n in range(len(fir_fr))]
    fir_fr_centroid_arr = np.vstack((fir_fr_centroid_x, fir_fr_centroid_y)).T

    sec_fr = np.load(os.path.join(bbox_dir, iter_bbox[i+1]))
    sec_fr_centroid_y = [np.average((sec_fr[n, 0], sec_fr[n, 2])) for n in range(len(sec_fr))]
    sec_fr_centroid_x = [np.average((sec_fr[n, 1], sec_fr[n, 3])) for n in range(len(sec_fr))]
    sec_fr_centroid_arr = np.vstack((sec_fr_centroid_x, sec_fr_centroid_y)).T

    timestamp = find_timestamp(iter_bbox[i+1])
    # img = plt.imread("/content/drive/MyDrive/data/hypoxia/17-07-18-04hypoxia8days_oxygenupatend.czi/preproc_img/17-07-18-04hypoxia8days_oxygenupatend.czi - 17-07-18-04hypoxia8days_oxygenupatend.czi #1-{} (dragged).tiff".format(timestamp))

    N = len(fir_fr_centroid_arr)
    M = len(sec_fr_centroid_arr)
    cost_mat = np.zeros(shape=(N, M)) # cost matrix that we will fill out

    for i in range(0,N):
      for j in range(0,M):
        cent_dist = distance.euclidean(fir_fr_centroid_arr[i], sec_fr_centroid_arr[j])
        cost_mat[i][j] = cent_dist

    row_ind, col_ind = linear_sum_assignment(cost_mat)

    for i in range(len(row_ind)):
      x_val = (fir_fr_centroid_arr[row_ind[i]][1], sec_fr_centroid_arr[col_ind[i]][1])
      y_val = (fir_fr_centroid_arr[row_ind[i]][0], sec_fr_centroid_arr[col_ind[i]][0])
    ## start here 
      point1 = (fir_fr_centroid_arr[row_ind[i]][1], fir_fr_centroid_arr[row_ind[i]][0])
      point2 = (sec_fr_centroid_arr[col_ind[i]][1], sec_fr_centroid_arr[col_ind[i]][0])
      traj_len = distance.euclidean(point1, point2)
      if traj_len < 200:
        plt.plot(x_val, y_val, c='b')
        frame_ids.append([row_ind[i], col_ind[i]])

    cell_id_match.append(frame_ids)
  return cell_id_match



def hungarian(bbox_dir):
  bbox_files = os.listdir(bbox_dir)
  for i in reversed(range(len(bbox_files)-1)):
    fir_fr = np.load(os.path.join(bbox_dir, bbox_files[i+1]))
    fir_fr_centroid_y = [np.average((fir_fr[n, 0], fir_fr[n, 2])) for n in range(len(fir_fr))]
    fir_fr_centroid_x = [np.average((fir_fr[n, 1], fir_fr[n, 3])) for n in range(len(fir_fr))]
    fir_fr_centroid_arr = np.vstack((fir_fr_centroid_x, fir_fr_centroid_y)).T

    sec_fr = np.load(os.path.join(bbox_dir, bbox_files[i]))
    sec_fr_centroid_y = [np.average((sec_fr[n, 0], sec_fr[n, 2])) for n in range(len(sec_fr))]
    sec_fr_centroid_x = [np.average((sec_fr[n, 1], sec_fr[n, 3])) for n in range(len(sec_fr))]
    sec_fr_centroid_arr = np.vstack((sec_fr_centroid_x, sec_fr_centroid_y)).T

    timestamp = find_timestamp(bbox_files[i])
    dist_dict[timestamp] = None
    img = plt.imread("/content/drive/MyDrive/mphil/preprocessed_xfer_data/17-07-18-04hypoxia8days_oxygenupatend/preproc_img/t:{}:480 - 17-07-18-04hypoxia8days_oxygenupatend.czi #1.tif".format(timestamp))

    N = len(fir_fr_centroid_arr)
    M = len(sec_fr_centroid_arr)
    cost_mat = np.zeros(shape=(N, M)) # cost matrix that we will fill out

    for i in range(0,N):
      for j in range(0,M):
        cent_dist = distance.euclidean(fir_fr_centroid_arr[i], sec_fr_centroid_arr[j])
        cost_mat[i][j] = cent_dist

    row_ind, col_ind = linear_sum_assignment(cost_mat)
    return row_ind, col_ind