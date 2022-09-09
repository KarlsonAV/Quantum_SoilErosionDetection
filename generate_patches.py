import os

import numpy as np
import cv2

from patchify import patchify

import rasterio
from rasterio.plot import reshape_as_image

import albumentations as A


raster_path = "Data/T36UXV_20200406T083559_TCI_10m.jp2"
mask_path = "Data/train.jp2"

with rasterio.open(raster_path, "r", driver="JP2OpenJPEG") as src:
    raster_img = src.read()
    raster_meta = src.meta

with rasterio.open(mask_path, "r", driver="JP2OpenJPEG") as src:
    mask_img = src.read()
    mask_meta = src.meta


raster_img = reshape_as_image(raster_img)
mask_img = reshape_as_image(mask_img)

# Original is 10980
IM_WIDTH = IM_HEIGHT = 10752
raster_img = cv2.resize(raster_img, (IM_HEIGHT, IM_WIDTH))
mask_img = cv2.resize(mask_img, (IM_HEIGHT, IM_WIDTH))
mask_img = np.expand_dims(mask_img, axis=2)

PATCH_WIDTH = 512
raster_patches = patchify(raster_img, (PATCH_WIDTH, PATCH_WIDTH, 3), PATCH_WIDTH)
mask_patches = patchify(mask_img, (PATCH_WIDTH, PATCH_WIDTH, 1), PATCH_WIDTH)


m, n, *_ = raster_patches.shape

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.GridDistortion(p=0.5)
])

augmentations = 8
for a in range(augmentations):

    augmentation_path = f"Data/patches/augmentation_{a}"
    os.makedirs(augmentation_path, exist_ok=True)
    os.makedirs(augmentation_path + "/images", exist_ok=True)
    os.makedirs(augmentation_path + "/masks", exist_ok=True)

    for i in range(m):
        for j in range(n):
            raster_patch = raster_patches[i, j, 0]
            mask_patch = mask_patches[i, j, 0]
            transformed = transform(image=raster_patch, mask=mask_patch)

            fname = f"patch_raster_{i}_{j}.png"
            cv2.imwrite(augmentation_path + "/images/" + fname, cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR))

            fname = f"patch_mask_{i}_{j}.png"
            cv2.imwrite(augmentation_path + "/masks/" + fname, cv2.cvtColor(transformed['mask'], cv2.COLOR_RGB2BGR))
