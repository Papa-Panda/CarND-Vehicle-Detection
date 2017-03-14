import cv2
import numpy as np

import matplotlib.image as mpimg

from skimage.feature import hog
from sklearn.model_selection import GridSearchCV

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def transform_colorspace(image, cspace='RGB'):
    feature_image = np.copy(image)

    if cspace == 'RBG':
        return feature_image

    if cspace == 'HSV':
        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    elif cspace == 'LUV':
        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    elif cspace == 'HLS':
        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    elif cspace == 'YUV':
        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    elif cspace == 'YCrCb':
        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

    return feature_image

def extract_feature(image, 
                      cspace='RGB', 
                      orient=9, 
                      pixels_per_cell=8, 
                      cell_per_block=2, 
                      hg_channel=0,
                      spatial_size=(32, 32), 
                      hist_bins=32):

    feature_image = transform_colorspace(image, cspace)
    spatial_features = bin_spatial(feature_image, size=spatial_size)
    hist_features = color_hist(feature_image, nbins=hist_bins)

    # Call get_hog_features() with vis=False, feature_vec=True
    if hg_channel == 'ALL':
        hog_features = []
        for channel in range(feature_image.shape[2]):
            hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                orient, pixels_per_cell, cell_per_block, 
                                vis=False, feature_vec=True))
        hog_features = np.ravel(hog_features)        
    else:
        hog_features = get_hog_features(feature_image[:,:,hg_channel], orient, 
                    pixels_per_cell, cell_per_block, vis=False, feature_vec=True)
    # Append the new feature vector to the features list
    return np.concatenate((hog_features, spatial_features, hist_features))

def extract_features(fnames, 
                     cspace='YCrCb', 
                     orient=9, 
                     pixels_per_cell=8, 
                     cell_per_block=2, 
                     hg_channel='ALL',
                     spatial_size=(32, 32), 
                     hist_bins=32):
    all_features = []

    for paths in fnames:
        features = []
        for file_loc in paths:
            image = mpimg.imread(file_loc)
            features.append(extract_feature(image, cspace, orient, pixels_per_cell,
                                            cell_per_block, hg_channel, spatial_size, 
                                            hist_bins))

        all_features.append(features)

    # Return list of feature vectors
    return all_features
