import os 
import json
from pycocotools.coco import COCO
from pycocotools import mask


from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon


from skimage import measure
from scipy import ndimage as ndi

annFile = "path/to/json"

coco = COCO(annFile)
image_id = 59
img = coco.imgs[image_id]
img_ids = coco.getImgIds()
cat_ids = coco.getCatIds()
anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
anns = coco.loadAnns(anns_ids) # load anno

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

        for n in range(len(anns)):
            mask += coco.annToMask(anns[n])
    
        im = Image.fromarray(mask)
        im.save(os.path.join(output_dir, img_fname), compression='deflate')

def procAnns(anns):
  """
  anns is the annotation file that you have to load to coco
  """
  annotations = []
  if len(anns) ==0:
    return 0 
  if 'segmentation' in anns[0]: 
    datasetType = 'instances'
  else:
    raise Exception('datasetType not supported')
  if datasetType == 'instances':
    polygons = []
    for ann in anns:
      if 'segmentation' in ann:
        if type(ann['segmentation']) == list:
          binary_mask = coco.annToMask(ann)
          distance_matrix = ndi.distance_transform_edt(binary_mask)
          for i in range(len(distance_matrix)):
            ind_low = np.where(distance_matrix[i] < 4)
            distance_matrix[i][ind_low] = 0
            ind_high = np.where(distance_matrix[i] >= 4)
            distance_matrix[i][ind_high] = 1
          
          # as uint8
          distance_matrix = distance_matrix.astype(np.uint8)

          fortran_ground_truth_binary_mask = np.asfortranarray(distance_matrix)
          encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
          ground_truth_area = mask.area(encoded_ground_truth)
          ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
          contours = measure.find_contours(distance_matrix, 0.5)

          for contour in contours:
            contour = np.flip(contour, axis=1)
            segmentation = contour.ravel().tolist()
          ann["segmentation"] = [segmentation]
  return anns # this will be all the annotations for each image_id

def visualise_anns(anns, draw_bbox=False):
        """
        Display the specified annotations same colour for each frame -> TODO change later
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return 0
        if 'segmentation' in anns[0] or 'keypoints' in anns[0]:
            datasetType = 'instances'
        elif 'caption' in anns[0]:
            datasetType = 'captions'
        else:
            raise Exception('datasetType not supported')
        if datasetType == 'instances':
            ax = plt.gca()
            ax.set_autoscale_on(False)
            polygons = []
            color = []
            c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
            for ann in anns:
                if 'segmentation' in ann:
                    if type(ann['segmentation']) == list:
                        # polygon
                        for seg in ann['segmentation']:
                            poly = np.array(seg).reshape((int(len(seg)/2), 2))
                            polygons.append(Polygon(poly))
                            color.append(c)

                if draw_bbox:
                    [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
                    poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
                    np_poly = np.array(poly).reshape((4,2))
                    polygons.append(Polygon(np_poly))
                    color.append(c)

            p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
            ax.add_collection(p)
            p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
            ax.add_collection(p)
        elif datasetType == 'captions':
            for ann in anns:
                print(ann['caption'])
