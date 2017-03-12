import cv2
import numpy as np
import glob
import time
import pickle

from train_vehicle_classifier import extract_features

from sklearn.preprocessing import StandardScaler

def fetch_data():
    import urllib
    import zipfile
    import os

    if os.path.exists('./data'):
        return

    os.mkdir('./data')
    vehicles = 'https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip'
    non_vehicles = 'https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip'

    urls = [vehicles, non_vehicles]

    print ("Fetching data....")
    for url in urls:
        path = "./data/%s"%(url.split('/')[-1])
        urllib.request.urlretrieve(url, path)
        zipfile.ZipFile(path, 'r').extractall('./data')

def load_path(path, train_perc=.70, valid_perc=.15, test_perc=.15, shuffle=False):
    '''
    Load images and split them into train/test/valid according to given percentages
    '''
    from random import shuffle

    imgs = glob.glob(path)
    if shuffle:
        shuffle(imgs)

    n = len(imgs)
    n_train = int(n * train_perc)
    n_valid = int(n * valid_perc)
    n_test = int(n * test_perc)

    return imgs[:n_train], imgs[n_train+1:n_train+1+n_valid], imgs[n_train+n_valid+2:]

def preprocess(path="./data"):
    gti_far = "%s/vehicles/GTI_Far/*.png"%(path)
    gti_left = "%s/vehicles/GTI_Left/*.png"%(path)
    gti_middle = "%s/vehicles/GTI_MiddleClose/*.png"%(path)
    gti_right = "%s/vehicles/GTI_Right/*.png"%(path)
    kiti = "%s/vehicles/KITTI_extracted/*.png"%(path)
    non_vehicles = "%s/non-vehicles/*/*.png"%(path)

    train_imgs = []
    valid_imgs = []
    test_imgs  = []

    paths = [gti_far, gti_left, gti_middle, gti_right]
    for path in paths:
        train, valid, test = load_path(path)
        train_imgs.extend(train)
        valid_imgs.extend(valid)
        test_imgs.extend(test)

    train, valid, test = load_path(kiti, shuffle=True)
    train_imgs.extend(train)
    valid_imgs.extend(valid)
    test_imgs.extend(test)

    colorspace = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = 0 # Can be 0, 1, 2, or "ALL"

    car_imgs = [train_imgs, valid_imgs, test_imgs]

    car_features = extract_features(car_imgs, cspace=colorspace, orient=orient, 
                        pixels_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hg_channel=hog_channel)

    train, valid, test = load_path(non_vehicles, shuffle=True)
    noncar_imgs = [train, valid, test]
    notcar_features = extract_features(noncar_imgs, cspace=colorspace, orient=orient, 
                        pixels_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hg_channel=hog_channel)

    print ("Car features in training ", len(car_features[0]))
    print ("Non-Car features in training ", len(notcar_features[0]))

    print ("Car features in validation ", len(car_features[1]))
    print ("Non-Car features in validation ", len(notcar_features[1]))

    print ("Car features in test ", len(car_features[2]))
    print ("Non-Car features in test ", len(notcar_features[2]))

    # Create an array stack of feature vectors
    X_train = np.vstack((car_features[0], notcar_features[0])).astype(np.float64)
    X_valid = np.vstack((car_features[1], notcar_features[1])).astype(np.float64) 
    X_test = np.vstack((car_features[2], notcar_features[2])).astype(np.float64) 

    # # Fit a per-column scaler
    X_train_scaler = StandardScaler().fit(X_train)
    scaled_X_train = X_train_scaler.transform(X_train)

    X_valid_scaler = StandardScaler().fit(X_valid)
    scaled_X_valid = X_valid_scaler.transform(X_valid)

    X_test_scaler = StandardScaler().fit(X_test)
    scaled_X_test = X_test_scaler.transform(X_test)

    # # Define the labels vector
    y_train = np.hstack((np.ones(len(car_features[0])), np.zeros(len(notcar_features[0]))))
    y_valid = np.hstack((np.ones(len(car_features[1])), np.zeros(len(notcar_features[1]))))
    y_test = np.hstack((np.ones(len(car_features[2])), np.zeros(len(notcar_features[2]))))

    data = {}
    data["X_train"] = scaled_X_train
    data["X_valid"] = scaled_X_valid
    data["X_test"] = scaled_X_test
    data["y_train"] = y_train
    data["y_valid"] = y_valid
    data["y_test"] = y_test

    with open('./data/vehicle_data.pickle', 'wb') as file:
        pickle.dump(data, file)


if __name__ == "__main__":
    fetch_data()
    preprocess()