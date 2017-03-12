import cv2
import numpy as np
import time
import pickle

import matplotlib.image as mpimg

from sklearn.svm import LinearSVC
from skimage.feature import hog
from sklearn.model_selection import GridSearchCV

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


def extract_features(fnames, cspace='RGB', orient=9, 
                     pixels_per_cell=8, cell_per_block=2, hg_channel=0):
    all_features = []

    for paths in fnames:
        features = []
        for file_loc in paths:
            image = mpimg.imread(file_loc)
            
            feature_image = np.copy(image)

            if cspace != 'RBG':
                if cspace == 'HSV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                elif cspace == 'LUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
                elif cspace == 'HLS':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
                elif cspace == 'YUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            
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
            features.append(hog_features)

        all_features.append(features)

    # Return list of feature vectors
    return all_features

def train():
    data = None
    with open('./data/vehicle_data.pickle', 'rb') as file:
        data = pickle.load(file)

    X_train = data['X_train']
    X_valid = data['X_valid']
    X_test = data['X_test']
    y_train = data['y_train']
    y_valid = data['y_valid']
    y_test = data['y_test']

    print ("Training data " , X_train.shape, y_train.shape)
    print ("Validation data ", X_valid.shape, y_test.shape)
    print ("Test data ", X_test.shape, y_valid.shape)

    param_grid = [
          {'C': [1, 10, 100, 1000], 'loss': ['hinge']},
          {'C': [1, 10, 100, 1000], 'loss': ['squared_hinge']},
    ]

    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')

    print('Validation Accuracy of SVC = ', round(svc.score(X_valid, y_valid), 4))

def main():

    train()
    
if __name__ == "__main__":
    main()
