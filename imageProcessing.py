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

def get_training_features(train_path, orient, pix_per_cell, cell_per_block, color_space = 'RGB', hog_channel = 'ALL'):
    features_list = []
    train_image_folders = os.listdir(train_path)
    for folder in train_image_folders:
        folder_path = train_path + "\\" + folder
        imgs = os.listdir(folder_path)
        for i in range(len(imgs)):
            img_path = folder_path + "\\" + imgs[i]
            #read image
            img = mpimg.imread(img_path)
            #extract features
            features = extract_features(img, orient, pix_per_cell, cell_per_block, color_space = 'RGB', hog_channel = 'ALL')
            features_list.append(np.concatenate(features))
    print(len(features_list))
    print(len(features_list[0]))
    return features_list

def extract_features(img, orient, pix_per_cell, cell_per_block, color_space = 'RGB', hog_channel = 'ALL'):
    features_list = []
    #convert to desired color_space
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)
    #extract HOG features and append to features list
    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(feature_image.shape[2]):
            features = get_hog_features(feature_image[:,:,channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            hog_features.append(features)
        hog_features = np.ravel(hog_features)
    else:
        features = get_hog_features(feature_image[:,:,hog_channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        hog_features.append(features)
    features_list.append(hog_features)
    return features_list
