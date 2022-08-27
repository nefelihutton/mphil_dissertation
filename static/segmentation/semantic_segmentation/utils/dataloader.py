import logging
import os
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
import cv2
from torch.utils.data import Dataset, DataLoader
import math

class BinaryDataset(Dataset):
    def __init__(self, img_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        # data loading
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.ids = [splitext(file)[0] for file in os.listdir(img_dir) if not file.startswith('.')] # so it doesn't read in .DSstore
        if not self.ids: 
            raise RuntimeError(f'No input file found in {img_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        # allow for len(dataset)
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, target_w, is_mask):
        # resize image & mask to 128
        # w, h = pil_img.size
        # target_w = 128
        # scale_ratio = w/128

        # newW, newH = int(1/scale_ratio * w), int(1/scale_ratio * h)
        # assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        # pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        # img_ndarray = np.asarray(pil_img)

        # un-comment for backbone
        w, h = pil_img.size
        target_w = 128
        w_scale_ratio = w/128
        h_scale_ratio = h/128

        newW, newH = int(1/w_scale_ratio * w), int(1/h_scale_ratio * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        # normalise pixel values
        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            img_ndarray = img_ndarray / 255

        return img_ndarray

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext in ['.tiff', '.tif']:
            return Image.open(filename)
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        # this allows for indexing later dataset[0]
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.img_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }