import numpy as np
from filterpy.kalman import KalmanFilter
from dprint import dprint
from scipy.optimize import linear_sum_assignment

from scipy.spatial import distance

import os
import matplotlib.pyplot as plt

from shape_analysis.edges import find_between
from shape_analysis.edges import find_timestamp

bbox_dir = "/content/drive/MyDrive/mphil/preprocessed_xfer_data/23-05-2018-exp1_normoxia_2min_x10_3pos/bbox_coords"
bbox_files = os.listdir(bbox_dir)

x_test = []
y_test = []

dist_dict = {}

def val_append(dict_obj, key, value):
 if key in dict_obj:
  if not isinstance(dict_obj[key], list):
  # converting key to list type
   dict_obj[key] = [dict_obj[key]]
   # Append the key's value in list
   dict_obj[key].append(value)

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

  plt.figure(figsize=(8, 8))

  x_ = []
  y_ = []
  dist_traved = []

  for i in range(len(row_ind)):
    x_val = (fir_fr_centroid_arr[row_ind[i]][1], sec_fr_centroid_arr[col_ind[i]][1])
    y_val = (fir_fr_centroid_arr[row_ind[i]][0], sec_fr_centroid_arr[col_ind[i]][0])
  ## start here 
    point1 = (fir_fr_centroid_arr[row_ind[i]][1], fir_fr_centroid_arr[row_ind[i]][0])
    point2 = (sec_fr_centroid_arr[col_ind[i]][1], sec_fr_centroid_arr[col_ind[i]][0])
    traj_len = distance.euclidean(point1, point2)
  # print(x_val, y_val, traj_len)
    if traj_len < 500:
      dist_traved.append(traj_len)
      # plt.plot(x_val, y_val, c='b')
      x_.append(x_val)
      y_.append(y_val)
    else:
      pass
  
  x_test.append(x_)
  y_test.append(y_)
  val_append(dist_dict,timestamp,dist_traved)

  # end here
  plt.scatter(fir_fr_centroid_arr[:,1], fir_fr_centroid_arr[:,0], c='k')
  plt.scatter(sec_fr_centroid_arr[:,1], sec_fr_centroid_arr[:,0], c='b')
  plt.imshow(img)
  plt.title("t={}".format(timestamp))

  plt.show()

# make fig
plt.figure(figsize=(8, 8))

for j in range(len(x_test)):
  for i in range(len(x_test[j])):
    plt.plot(x_test[j][i], y_test[j][i], c='b')
plt.ylim((2100, 0))
plt.title("All trajectories")
plt.show()

speed = []
for i in range(29):
  if i == 0:
    dists = np.asarray(list(dist_dict.values())[i][1])
    time = int(list(dist_dict.keys())[i])
    time = 0.4 * time
    speed.append(np.around(dists/time))
  else:
    dists = np.asarray(list(dist_dict.values())[i][1])
    time = int(list(dist_dict.keys())[i]) - int(list(dist_dict.keys())[i-1])
    time = 0.4 * time
    speed.append(np.around(dists/time))

speed_arr = np.hstack(speed)
num = [len(np.where(speed_arr == np.unique(speed_arr)[i])[0]) for i in range(len(np.unique(speed_arr)))]
speed_hist = np.vstack((np.unique(speed_arr), num)).T

fig, ax = plt.subplots(1, 1)
ax.grid(zorder=0)
ax.bar(speed_hist[:, 0], speed_hist[:, 1], width=1, align='center', color='k', zorder=5)
plt.title('Mean speed, hypoxia')
plt.xlabel("Mean speed (μm/hr)")
plt.xlim((0, 250))
plt.ylabel("Number of cells")
plt.show()