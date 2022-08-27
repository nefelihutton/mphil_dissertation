import os 
import numpy as np
from PIL import Image

dir = "LIVECELL_binary_fixed"
parent_dir = "/content/drive/MyDrive/segmentation"
os.mkdir(os.path.join(parent_dir, dir))

livecell_dir = "/content/drive/MyDrive/segmentation/LIVECell_binary_masks"
output_dir = "/content/drive/MyDrive/segmentation/LIVECELL_binary_fixed"

for mask in os.listdir(livecell_dir):
  np_mask = np.asarray(Image.open(os.path.join(livecell_dir, mask)))
  np_mask = np_mask.copy()
  np_mask[np_mask > 0] = 1
  im = Image.fromarray(np_mask)
  im.save(str(os.path.join(output_dir, mask)))