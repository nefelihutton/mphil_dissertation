import numpy as np
from scipy.signal import find_peaks



def get_peaks(shape_signature_dict):
  arm_no_dict = {}
  arm_size_dict = {}
  for frame in range(len(shape_signature_dict.items())):
    extension_number = []
    extensions_size = []
    a = list(shape_signature_dict.values())[frame]
    for idx in range(len(a)):
    # this processes one shape
      idx # shape id 
      if (np.max(a[idx][:,1]) - np.average(a[idx][:,1])) < 15:
        number_prot = 0
        prot_size = 0
        extension_number.append(number_prot)
        # extensions_size.append(prot_size)
      else: 
        thresh = np.average(a[idx][:,1])
        peak, _ = find_peaks(a[idx][:,1], height = np.average(a[idx][:,1]), prominence=1)
        number_prot = len(peak)
        prot_size = a[idx][:,1][peak].tolist() # this is an array of all the y values
        extension_number.append(number_prot)
        extensions_size.append(prot_size)
    all_arm_size = [x for xs in extensions_size for x in xs]

    time = int(list(shape_signature_dict.keys())[frame])
    arm_no_dict[time] = extension_number
    arm_size_dict[time] = all_arm_size
  return arm_no_dict, arm_size_dict