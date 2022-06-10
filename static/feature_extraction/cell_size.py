import numpy as np
import alphashape
from edges import flatten

def find_mask_area(seg_mask):
  """
  Input: segm_mask output from inference
  returns 2d edge coordinates of the inferred masks
  """
  area_list = []
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
      shape_area = alpha_shape.area
    else:
      shape_area = np.array([x.area for x in alpha_shape.geoms])      
    area_list.append(shape_area)

  return list(flatten(area_list))