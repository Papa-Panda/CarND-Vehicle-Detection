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

from moviepy.editor import VideoFileClip

class Tracker():
    def __init__(self):
        self.heatmap = None
        self.ystart = 400
        self.ystop = 656
        self.scales = [1.0, 1.2, 1.5]
        
        self.orient = 9
        self.pix_per_cell = 8
        self.cell_per_block = 2
        self.spatial_size = (32, 32)
        self.hist_bins = 32
        self.labels = None

        # Number of pixels should belong to a car
        self.threshold = 5
        self.frame = 0
        self.skipFrmaes = 0
        self.lastFrame = -1 # -1 will process all frames. 

        data = None
        with open('./data/svc_vehicle_classifier.pickle', 'rb') as file:
            data = pickle.load(file)

        self.svc = data['svc']
        self.X_scaler = data['X_scaler']

    # Define a single function that can extract features using hog sub-sampling and make predictions
    def find_cars(self, img, scale):
        
        ystart = self.ystart
        ystop = self.ystop
        svc = self.svc
        X_scaler = self.X_scaler
        orient = self.orient
        pix_per_cell = self.pix_per_cell
        cell_per_block = self.cell_per_block
        spatial_size = self.spatial_size
        hist_bins = self.hist_bins

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
        # imcopy = np.copy(img)

        for box in boxes:
            cv2.rectangle(img, box[0], box[1], color, thick)

        return img

    def update_heatmap(self, bbox):
        heatmap = self.heatmap

        for box in bbox:
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    def threshold_heatmap(self):
        heatmap = self.heatmap
        threshold = self.threshold        

        heatmap[heatmap <= threshold] = 0

    def draw_labeled_bboxes(self, img, labels):
        ystart = self.ystart

        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()

            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            # topX = np.min(nonzerox)
            # topY = np.min(nonzeroy)
            # botX = np.max(nonzerox)
            # botY = np.max(nonzeroy)

            cx = nonzero[1].mean()
            cy = nonzero[0].mean()
            
            # TODO Make these more configurable
            # Adjust the width of the window based on the detection location. 
            scale = (cy - ystart)/250.0
            win_size = 100 + (200 * scale)

            bbox = ( (int(cx-win_size/2), int(cy-win_size/2)), (int(cx+win_size/2), int(cy+win_size/2)) )
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)

        return img
        
    def update(self, img):

        # if self.frame < self.skipFrmaes or self.frame > self.lastFrame:
        #     self.frame += 1
        #     return img

        if self.heatmap == None:
            self.heatmap = np.zeros_like(img[:, :, 0]).astype(np.float)

        if self.frame != 0 and (self.frame % 3) != 0 :
            img = self.draw_labeled_bboxes(img, self.labels)
            self.heatmap[self.heatmap > 0 ] = 0
            self.frame += 1
            return img

        scales = [1.0, 1.2, 1.3, 1.5]
        bbox = []
        for scale in scales:
            bbox.extend(self.find_cars(img, scale))

        self.update_heatmap(bbox)
        self.threshold_heatmap()
        self.labels = label(self.heatmap)
        self.frame += 1

        return self.draw_labeled_bboxes(img, self.labels)

if __name__ == "__main__":
    import sys

    tracker = Tracker()
    tracker.skipFrmaes = 4 * 25
    tracker.lastFrame = 8 * 25

    clip1 = VideoFileClip(sys.argv[1])
    video_clip = clip1.fl_image(tracker.update)
    video_clip.write_videofile('out.mp4', audio=False)

    # img = mpimg.imread('test_images/test1.jpg')
    # out = tracker.update(img)

    # plt.imshow(out)
    # plt.show()


