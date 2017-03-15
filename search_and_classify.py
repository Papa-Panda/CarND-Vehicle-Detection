'''
A set of experiments. The main entry point is tracker.py
'''

import numpy as np
import cv2
import pickle

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from features import get_hog_features
from features import bin_spatial
from features import color_hist
from features import convert_color

from scipy.ndimage.measurements import label

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    img = img.astype(np.float32)/255
    
    result = []
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step

            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            tmp = np.hstack((hog_features, spatial_features, hist_features))
            test_features = X_scaler.transform(np.hstack((hog_features, spatial_features, hist_features)).reshape(1, -1))
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                result.append([(xbox_left, ytop_draw+ystart), (xbox_left+win_draw, ytop_draw+win_draw+ystart)])
                
    return result

def draw_boxes(img, boxes, color=(0, 0, 255), thick=5):
    imcopy = np.copy(img)

    for box in boxes:
        cv2.rectangle(imcopy, box[0], box[1], color, thick)

    return imcopy

def add_heat(heatmap, bbox):
    for box in bbox:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    return heatmap

def apply_threshold(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0
    return heatmap

def draw_labeled_bboxes(img, labels):
    print (labels[0].shape)


    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()

        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        topX = np.min(nonzerox)
        topY = np.min(nonzeroy)
        botX = np.max(nonzerox)
        botY = np.max(nonzeroy)

        cx = nonzero[1].mean()
        cy = nonzero[0].mean()
        
        # TODO Make these more configurable
        # Adjust the width of the window based on the detection location. 
        scale = (cy - 400)/150.0
        win_size = 100 + (150 * scale)

        # Define a bounding box based on min/max x and y
        # bbox = ((topX, topY), (botX, botY))
        # cv2.rectangle(img, bbox[0], bbox[1], (0,255,0), 6)

        bbox = ( (int(cx-win_size/2), int(cy-win_size/2)), (int(cx+win_size/2), int(cy+win_size/2)) )
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)

    # Return the image
    return img

def process():
    import glob

    ystart = 400
    ystop = 656
    scales = [1.0, 1.2, 1.3, 1.5]
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    spatial_size = (32, 32)
    hist_bins = 32

    data = None
    with open('./data/svc_vehicle_classifier.pickle', 'rb') as file:
        data = pickle.load(file)

    svc = data['svc']
    X_scaler = data['X_scaler']

    f, ax = plt.subplots(6, 4, figsize=(50, 50))
    # img = mpimg.imread('test_images/test1.jpg')
    i = 0
    fnames = glob.glob('test_images2/*.jpg')
    for fname in fnames:
        print (i)
        img = mpimg.imread(fname)

        heatmap = np.zeros_like(img[:, :, 0]).astype(np.float)

        bbox = []
        for scale in scales:
            bbox.extend(find_cars(img, ystart, ystop, scale, svc,
                        X_scaler, orient, pix_per_cell, 
                        cell_per_block, spatial_size, hist_bins))

        # ax[i, 0].imshow(img)
        # if i == 0:
        #     ax[i, 0].set_title('Original Image', fontsize=30)

        out_img = draw_boxes(img, bbox)
        ax[i, 0].imshow(out_img)
        if i == 0:
            ax[i, 0].set_title('Hog Detection', fontsize=30)
        
        heatmap = add_heat(heatmap, bbox)
        heatmap = apply_threshold(heatmap, 5)

        heatplot = np.clip(heatmap, 0, 255)
        ax[i, 1].imshow(heatplot, cmap='hot')
        if i == 0:
            ax[i, 1].set_title('Heat Map', fontsize=30)
        
        labels = label(heatmap)
        ax[i, 2].imshow(labels[0], cmap='gray')
        if i == 0:
            ax[i, 2].set_title('Labels', fontsize=30)

        draw_img = draw_labeled_bboxes(np.copy(img), labels)

        ax[i, 3].imshow(draw_img)
        if i == 0:
            ax[i, 3].set_title('Result', fontsize=30)

        i += 1

    plt.savefig('output_images/pipeline_video_frame.jpg')

    #     import glob
    # i = 0
    # fm = 0
    # f, ax = plt.subplots(6, 4, figsize=(50, 50))
    # fnames = glob.glob('test_images2/test*.jpg')
    # for fname in fnames:
    #     img = mpimg.imread(fname)
    #     img_out = tracker.update(img)

    #     ax[i, 0].imshow(tracker.tmp_img)
    #     if i == 0:
    #         ax[i, 0].set_title('Hog Detection', fontsize=30)

    #     heatplot = np.clip(tracker.heatmap, 0, 255)
    #     ax[i, 1].imshow(heatplot, cmap='hot')
    #     if i == 0:
    #         ax[i, 1].set_title('Heat Map Threshold', fontsize=30)

    #     ax[i, 2].imshow(tracker.labels[0], cmap='gray')
    #     if i == 0:
    #         ax[i, 2].set_title('Labels', fontsize=30)

    #     ax[i, 3].imshow(img_out)
    #     if i == 0:
    #         ax[i, 3].set_title('Result', fontsize=30)

    #     print (i, tracker.frame)
    #     i += 1

    # plt.savefig('output_images/pipeline_video_frame.jpg')

    # img = mpimg.imread('test_images/test1.jpg')
    # out = tracker.update(img)

    # plt.imshow(out)
    # plt.show()

if __name__ == "__main__":
    process()
