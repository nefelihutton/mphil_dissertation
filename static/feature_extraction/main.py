import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

import alphashape

import math
from math import degrees, atan2
from scipy.signal import find_peaks

from edges import mask_to_edges, get_distance_centroid, clockwise_angle
from peaks import find_peak_dets

cell_shape_df = pd.DataFrame(columns = ['timestamp','number_of_protrusions', 'protrusion_pixel_length'])


for f in range(len(seg_coords_dir)):
  # get timestamp
  fname = seg_coords_files[f]
  timestamp = find_timestamp(str(fname))

  # load np array seg coords inference
  seg_coords = np.load(os.path.join(seg_coords_dir, seg_coords_files[f]))
  # mask_to_edges takes a file of detected mask output and returns an array of mask_edges
  mask_edges = mask_to_edges(seg_coords)

  shapes = []

  for num in range(len(mask_edges)):
    if len(np.shape(mask_edges[num])) is 2: 
      shape_sig = get_distance_centroid(mask_edges[num])
      shapes.append(shape_sig)
      peaks, lengths = find_peak_dets(shapes)
      for val in range(len(peak)):
        cell_shape_df = cell_shape_df.append({'timestamp': timestamp, 
                                            'number_of_protrusions': peak[val],
                                            'protrusion_pixel_length': length[val]}, 
                                            ignore_index=True)
    else: 
      print(fname)