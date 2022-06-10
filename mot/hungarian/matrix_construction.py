import numpy as np
import os 
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

# set path
bbox_dir = "/content/drive/MyDrive/mphil/preprocessed_xfer_data/17-07-18-04hypoxia8days_oxygenupatend/bbox_coords"
preproc_img = "/content/drive/MyDrive/mphil/preprocessed_xfer_data/17-07-18-04hypoxia8days_oxygenupatend/preproc_img"
bbox_files = os.listidr(bbox_dir)

x_test = []
y_test = []


for i in reversed(range(len(bbox_files)-20)):
  first_frame = np.load(os.path.join(bbox_dir, bbox_files[i]))
  next_frame = np.load(os.path.join(bbox_dir, bbox_files[i+1]))
  # first frame 
  first_centroid_y = [np.average((first_frame[n, 0], first_frame[n, 2])) for n in range(len(first_frame))]
  first_centroid_x = [np.average((first_frame[n, 1], first_frame[n, 3])) for n in range(len(first_frame))]
  first_centroid_arr = np.vstack((first_centroid_x, first_centroid_y)).T
  # next frame
  next_centroid_y = [np.average((next_frame[m, 0], next_frame[m, 2])) for m in range(len(next_frame))]
  next_centroid_x = [np.average((next_frame[m, 1], next_frame[m, 3])) for m in range(len(next_frame))]
  next_centroid_arr = np.vstack((next_centroid_x, next_centroid_y)).T
  
  # empty matrix
  N = len(next_centroid_arr)
  M = len(first_centroid_arr)
  cost_mat = np.zeros(shape=(N, M))

  # cost matrix
  for f in range(0,N):
    for j in range(0,M):
      cent_dist = distance.euclidean(next_centroid_arr[f], first_centroid_arr[j])
      cost_mat[f][j] = cent_dist
  
  # hungarian algo
  row_ind, col_ind = linear_sum_assignment(cost_mat)

  x_ = []
  y_ = []
  for x in range(len(row_ind)):
    x_val = (next_centroid_arr[row_ind[x]][0], first_centroid_arr[col_ind[x]][0])
    y_val = (next_centroid_arr[row_ind[x]][1], first_centroid_arr[col_ind[x]][1])
  #   plt.plot(x_val, y_val)
  #   plt.scatter(x_val, y_val)
  # plt.show()
    x_.append(x_val)
    y_.append(y_val)
  
  x_test.append(x_)
  y_test.append(y_)
  # plt.scatter(next_centroid_arr[:,0], next_centroid_arr[:,1])

# plt.show()

for j in range(len(x_test)):
  for i in range(len(x_test[j])):
    plt.plot(x_test[j][i], y_test[j][i])
plt.show()