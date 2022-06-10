import os 
import json
from pycocotools.coco import COCO
from pycocotools import mask


from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

def coco_to_binary(annFile, output_dir):
    """
    annFile is the path to the COCO annotation file in .json format 
    the binary mask will be saved to the output_dir
    """
    coco = COCO(annFile) # read in COCO annotation file

    img_ids = coco.getImgIds() # get image ids from the COCO file
    cat_ids = coco.getCatIds() # get category id

    for x, i in enumerate(img_ids):
        img = coco.imgs[i] # load one image 

        img_info = coco.loadImgs([img_ids[x]])[0]
        img_fname = img_info["file_name"] # get the image file name for later 

        anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None) # get anno
        anns = coco.loadAnns(anns_ids) # load anno

        mask = coco.annToMask(anns[0])

        for n in range(len(anns)):
            mask = coco.annToMask(anns[n])
            im = Image.fromarray(mask)
            im.save(os.path.join(output_dir, img_fname+"_mask_{}".format(n)+".tif"), compression='deflate')