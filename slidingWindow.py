import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from imageProcessing import *

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    '''bboxes should be of the form: [((x1, y1), (x2, y2)), ((,),(,)), ...]'''
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates.
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], xy_window=(64,64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = int(img.shape[0]/2)
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]

    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]

    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))

    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)

    # Initialize a list to append window positions to
    window_list = []

    # Loop through finding x and y window positions
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

def search_windows(img, clf, windows, scaler, orient, pix_per_cell, cell_per_block, color_space = 'RGB', spatial_size=(32, 32), hist_bins=32, hog_channel = 'ALL', spatial_feat=True, hist_feat=True, hog_feat=True):
    on_windows = []
    #iterate over all the windows
    for window_size in windows:
        for window in window_size:
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            # test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)
            #Extract features for that window
            features = np.concatenate(extract_features(test_img, orient, pix_per_cell, cell_per_block, color_space, spatial_size, hist_bins, hog_channel, spatial_feat, hist_feat, hog_feat))
            #Scale extracted features to be fed to classifier
            test_features = scaler.transform(np.array(features).reshape(1, -1))
            #Predict using classifier
            prediction = clf.predict(test_features)
            #If prediction is positive, store window
            if prediction:
                on_windows.append(window)
    return on_windows
