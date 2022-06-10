import numpy as np
from scipy.signal import find_peaks

def find_peak_dets(shape_signature):
  number_of_protrusions = []
  protrusion_pixel_length = []

  for i in range(len(shape_signature)):
    radial_dist = shape_signature[i]
    thresh = np.average(radial_dist[:,1])
    peaks, _ = find_peaks(radial_dist[:,1], distance = thresh)

    number_prot = len(radial_dist[:,1][peaks])
    number_of_protrusions.append(number_prot)
    peak_heights = radial_dist[:,1][peaks]
    protrusion_pixel_length.append(peak_heights)

  return number_of_protrusions, protrusion_pixel_length


# def protrusions_count(shape_signatures):
#   number_of_protrusions = []
#   protrusion_pixel_length = []

#   for i in range(len(shape_signatures)):
#     radial_dist = shape_signatures[i]
#     thresh = np.average(radial_dist[:,1])
#     peaks, _ = find_peaks(radial_dist[:,1], distance = thresh)
#     number_prot = len(radial_dist[:,1][peaks])
#     number_of_protrusions.append(number_prot)
#     peak_heights = radial_dist[:,1][peaks]
#     protrusion_pixel_length.append(peak_heights)
    
#   return number_of_protrusions, protrusion_pixel_length