import numpy as np
from skimage.feature import hog
import cv2
import os
import matplotlib.image as mpimg

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        # Use skimage.hog() to get both features and a visualization
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    else:
        # Use skimage.hog() to get features only
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), visualise=vis, feature_vector=feature_vec)
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

def get_training_features(train_files, orient, pix_per_cell, cell_per_block, color_space = 'RGB', spatial_size=(32, 32), hist_bins=32, hog_channel = 'ALL', spatial_feat=True, hist_feat=True, hog_feat=True):
    features_list = []
    for img_path in train_files:
        #read image
        img = cv2.imread(img_path)
        #extract features
        features = extract_features(img, orient, pix_per_cell, cell_per_block, color_space, spatial_size, hist_bins, hog_channel, spatial_feat, hist_feat, hog_feat)
        features_list.append(np.concatenate(features))
    return features_list

def extract_features(img, orient, pix_per_cell, cell_per_block, color_space = 'RGB', spatial_size=(32, 32), hist_bins=32, hog_channel = 'ALL', spatial_feat=True, hist_feat=True, hog_feat=True):
    features_list = []
    #convert to desired color_space
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    else: feature_image = np.copy(img)

    #extract spatial binning features
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        features_list.append(spatial_features)

    if hist_feat == True:
        # Apply color_hist()
        hist_features = color_hist(feature_image, nbins=hist_bins)
        features_list.append(hist_features)

    if hog_feat == True:
        #extract HOG features and append to features list
        hog_features = []
        if hog_channel == 'ALL':
            for channel in range(feature_image.shape[2]):
                features = get_hog_features(feature_image[:,:,channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)
                hog_features.append(features)
            hog_features = np.ravel(hog_features)
        else:
            features = get_hog_features(feature_image[:,:,hog_channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            hog_features.append(features)
        features_list.append(hog_features)
    return features_list
