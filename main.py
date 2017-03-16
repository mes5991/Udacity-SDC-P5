import time
import sys
import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC
from imageProcessing import *
from slidingWindow import *

options = sys.argv

if "test_hog" in options:
    image_folder_far =  r'C:\Users\mes59\Documents\Udacity\SDC\Term 1\Project 5\Training Data\vehicles\GTI_Far'
    image_folder_left =  r'C:\Users\mes59\Documents\Udacity\SDC\Term 1\Project 5\Training Data\vehicles\GTI_Left'
    image_folder_middle =  r'C:\Users\mes59\Documents\Udacity\SDC\Term 1\Project 5\Training Data\vehicles\GTI_MiddleClose'
    image_folder_right =  r'C:\Users\mes59\Documents\Udacity\SDC\Term 1\Project 5\Training Data\vehicles\GTI_Right'
    image_folder_kitti =  r'C:\Users\mes59\Documents\Udacity\SDC\Term 1\Project 5\Training Data\vehicles\KITTI_extracted'
    image_folders = [image_folder_far, image_folder_left, image_folder_middle, image_folder_right, image_folder_kitti]
    for folder in image_folders:
        imgs = os.listdir(folder)
        for i in range(3):
            ind = np.random.randint(0, len(imgs))
            img_path = folder + '\\' + imgs[ind]
            img = mpimg.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Define HOG parameters
            orient = 9
            pix_per_cell = 8
            cell_per_block = 2

            features, hog_image = get_hog_features(gray, orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)

            plt.subplot(121)
            plt.imshow(img, cmap='gray')
            plt.title('Example Car Image')
            plt.subplot(122)
            plt.imshow(hog_image, cmap='gray')
            plt.title('HOG Visualization')
            plt.show()

if 'test_window' in options:
    test_imgs_path = r'C:\Users\mes59\Documents\Udacity\SDC\Term 1\Project 5\CarND-Vehicle-Detection-master\test_images'
    test_imgs = [f for f in glob.glob(test_imgs_path + '/*.jpg', recursive=False)]
    print(len(test_imgs))
    for test_img in test_imgs:
        img = cv2.imread(test_img)
        small_windows = slide_window(img, x_start_stop=[None, None], y_start_stop=[380, 450],
                        xy_window=(32,32), xy_overlap=(.5,.5))
        med_windows = slide_window(img, x_start_stop=[None, None], y_start_stop=[None, 550],
                        xy_window=(150,150), xy_overlap=(.75,.75))
        big_windows = slide_window(img, x_start_stop=[None, None], y_start_stop=[380, img.shape[0]],
                        xy_window=(256,256), xy_overlap=(0.9, .8))
        img = draw_boxes(img, big_windows, color=(255, 0, 0), thick=6)
        img = draw_boxes(img, med_windows, color=(0, 255, 0), thick=3)
        img = draw_boxes(img, small_windows, color=(0, 0, 255), thick=1)

        plt.imshow(img)
        plt.show()


if 'pipeline' in options:
    # Define festure extraction parameters
    orient = 9 #HOG
    pix_per_cell = 8 #HOG
    cell_per_block = 2 #HOG
    hog_channel = 'ALL' #HOG
    color_space = 'HLS'
    spatial_size = (32,32) #spatial binning
    hist_bins = 32 #color channel histograms
    spatial_feat = True
    hist_feat = True
    hog_feat = True

    #Get image paths
    car_train_path = r'C:\Users\mes59\Documents\Udacity\SDC\Term 1\Project 5\Training Data\vehicles'
    not_car_train_path = r'C:\Users\mes59\Documents\Udacity\SDC\Term 1\Project 5\Training Data\non-vehicles'
    vehicle_files = [f for f in glob.glob(car_train_path + '/**/*.png', recursive=False)]
    non_vehicle_files = [f for f in glob.glob(not_car_train_path + '/**/*.png', recursive=False)]

    #Extract training features. Currently just hog features.
    print("Extracting features from training data...")
    t = time.time()
    car_features = get_training_features(vehicle_files, orient, pix_per_cell, cell_per_block, color_space = color_space, spatial_size=spatial_size, hist_bins=hist_bins, hog_channel = hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
    not_car_features = get_training_features(non_vehicle_files, orient, pix_per_cell, cell_per_block, color_space = color_space, spatial_size=spatial_size, hist_bins=hist_bins, hog_channel = hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
    t2 = time.time()
    print("Feature extraction took ", round(t2 - t, 2), "seconds")

    #Normalize features
    # Create an array stack; StandardScaler() expects np.float64
    X = np.vstack((car_features, not_car_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    #Define labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(not_car_features))))

    #Split data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

    #Train linear SVC
    print("Training classifier...")
    svc = LinearSVC()
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print("Classifier training took ", round(t2 - t, 2), "seconds")

    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    test_imgs_path = r'C:\Users\mes59\Documents\Udacity\SDC\Term 1\Project 5\CarND-Vehicle-Detection-master\test_images'
    test_imgs = [f for f in glob.glob(test_imgs_path + '/*.jpg', recursive=False)]

    for test_img in test_imgs:
        img = cv2.imread(test_img)
        small_windows = slide_window(img, x_start_stop=[None, None], y_start_stop=[380, 450],
                        xy_window=(32,32), xy_overlap=(.5,.5))
        med_windows = slide_window(img, x_start_stop=[None, None], y_start_stop=[None, 550],
                        xy_window=(150,150), xy_overlap=(.75,.75))
        big_windows = slide_window(img, x_start_stop=[None, None], y_start_stop=[380, img.shape[0]],
                        xy_window=(256,256), xy_overlap=(0.9, .8))
        windows = [small_windows, med_windows, big_windows]
        hot_windows = search_windows(img, svc, windows, X_scaler, orient, pix_per_cell, cell_per_block, color_space = color_space, spatial_size=spatial_size, hist_bins=hist_bins, hog_channel = hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
        window_img = draw_boxes(img, hot_windows)
        window_img = cv2.cvtColor(window_img, cv2.COLOR_BGR2RGB)
        plt.imshow(window_img)
        plt.show()
