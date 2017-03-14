import numpy as np
import cv2
import glob
import random

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from setup import fetch_data

from features import get_hog_features
from features import extract_features
from features import extract_feature
from features import convert_color

from search_and_classify import *

# fetch_data()

car_imgs = glob.glob('./data/vehicles/*/*.png')
noncar_imgs = glob.glob('./data/non-vehicles/*/*.png')

idx1 = np.random.randint(len(car_imgs))
idx2 = np.random.randint(len(noncar_imgs))

car = mpimg.imread(car_imgs[idx1])
noncar = mpimg.imread(noncar_imgs[idx2])

# plt.imshow(car)
# plt.savefig('output_images/car1.jpg')

# plt.imshow(noncar)
# plt.savefig('output_images/noncar1.jpg')

# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
# ax1.imshow(car)
# ax1.set_title('Vehicle', fontsize=30)
# ax2.imshow(noncar)
# ax2.set_title('Non-Vehicle', fontsize=30)

# plt.savefig('output_images/car_nocar.jpg')

colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 0


carcopy= np.copy(car)
noncarcopy = np.copy(noncar)

carcopy = convert_color(carcopy, 'RGB2YCrCb')
noncarcopy = convert_color(noncarcopy, 'RGB2YCrCb')

hog_features, hog_img1 = get_hog_features(carcopy[:, :, 0], orient, pix_per_cell, cell_per_block, vis=True)
hog_features, hog_img2 = get_hog_features(carcopy[:, :, 1], orient, pix_per_cell, cell_per_block, vis=True)
hog_features, hog_img3 = get_hog_features(carcopy[:, :, 2], orient, pix_per_cell, cell_per_block, vis=True)

hog_features, hog_img4 = get_hog_features(noncarcopy[:, :, 0], orient, pix_per_cell, cell_per_block, vis=True)
hog_features, hog_img5 = get_hog_features(noncarcopy[:, :, 1], orient, pix_per_cell, cell_per_block, vis=True)
hog_features, hog_img6 = get_hog_features(noncarcopy[:, :, 2], orient, pix_per_cell, cell_per_block, vis=True)


f, ax = plt.subplots(3, 4, figsize=(20, 20))


ax[0,0].imshow(carcopy[:, :, 0], cmap='gray')
ax[0,0].set_title('Car-Ch1', fontsize=30)
ax[0,1].imshow(hog_img1, cmap='gray')
ax[0,1].set_title('Car-Ch1 Hog', fontsize=30)
ax[0,2].imshow(noncarcopy[:, :, 0], cmap='gray')
ax[0,2].set_title('Non-Car-Ch1', fontsize=30)
ax[0,3].imshow(hog_img4, cmap='gray')
ax[0,3].set_title('Non-Car-Ch1 Hog', fontsize=30)

ax[1,0].imshow(carcopy[:, :, 1], cmap='gray')
ax[1,0].set_title('Car-Ch2', fontsize=30)
ax[1,1].imshow(hog_img2, cmap='gray')
ax[1,1].set_title('Car-Ch2 Hog', fontsize=30)
ax[1,2].imshow(noncarcopy[:, :, 1], cmap='gray')
ax[1,2].set_title('Non-Car-Ch2', fontsize=30)
ax[1,3].imshow(hog_img5, cmap='gray')
ax[1,3].set_title('Non-Car-Ch2 Hog', fontsize=30)

ax[2,0].imshow(carcopy[:, :, 2], cmap='gray')
ax[2,0].set_title('Car-Ch2', fontsize=30)
ax[2,1].imshow(hog_img3, cmap='gray')
ax[2,1].set_title('Car-Ch2 Hog', fontsize=30)
ax[2,2].imshow(noncarcopy[:, :, 2], cmap='gray')
ax[2,2].set_title('Non-Car-Ch2', fontsize=30)
ax[2,3].imshow(hog_img6, cmap='gray')
ax[2,3].set_title('Non-Car-Ch2 Hog', fontsize=30)

plt.savefig('output_images/hog_features.jpg')


# hog_features, hog_img = get_hog_features(noncar[:, :, 0], orient, pix_per_cell, cell_per_block, vis=True)
# plt.imshow(hog_img, cmap='gray')
# plt.savefig('output_images/noncar1_hog.jpg')

# features = extract_feature(car, colorspace, orient, pix_per_cell, cell_per_block, hog_channel)
# print (features.shape)