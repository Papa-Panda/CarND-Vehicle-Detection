import cv2
import numpy as np
import time
import pickle

from sklearn.svm import LinearSVC


def train():
    data = None
    with open('./data/vehicle_data.pickle', 'rb') as file:
        data = pickle.load(file)

    # All data has already be normalized using StandardScaler. 
    X_train = data['X_train']
    X_valid = data['X_valid']
    X_test = data['X_test']
    y_train = data['y_train']
    y_valid = data['y_valid']
    y_test = data['y_test']
    X_scaler = data['X_scaler']


    print ("Training data " , X_train.shape, y_train.shape)
    print ("Validation data ", X_valid.shape, y_test.shape)
    print ("Test data ", X_test.shape, y_valid.shape)

    param_grid = [
          {'C': [1, 10, 100, 1000], 'loss': ['hinge']},
          {'C': [1, 10, 100, 1000], 'loss': ['squared_hinge']},
    ]

    svc = LinearSVC(C=1.0, loss='hinge')
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')

    print('Validation Accuracy of SVC = ', round(svc.score(X_valid, y_valid), 4))
    print('Test Accuracy of SVC = ', round(svc.score(X_valid, y_valid), 4))

    data = {}
    data['svc'] = svc
    data['X_scaler'] = X_scaler
    with open('./data/svc_vehicle_classifier.pickle', 'wb') as file:
        pickle.dump(data, file)

def main():

    train()
    
if __name__ == "__main__":
    main()
