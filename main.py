import time
import sys
import numpy as np
import cv2
import os
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
    test_img = r'C:\Users\mes59\Documents\Udacity\SDC\Term 1\Project 5\CarND-Vehicle-Detection-master\test_images\test1.jpg'
    img = mpimg.imread(test_img)
    small_windows = slide_window(img, x_start_stop=[None, None], y_start_stop=[int(img.shape[0]/2), 500],
                    xy_window=(64,64), xy_overlap=(0.5, 0.5))
    med_windows = slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(192,192), xy_overlap=(0.75, 0.75))
    big_windows = slide_window(img, x_start_stop=[None, None], y_start_stop=[0, img.shape[0]],
                    xy_window=(256,256), xy_overlap=(0.75, 0.75))
    window_img = draw_boxes(img, small_windows, color=(0, 0, 255))
    window_img = draw_boxes(window_img, med_windows, color=(0, 255, 0))
    window_img = draw_boxes(window_img, big_windows, color=(255, 0, 0))
    plt.imshow(window_img)
    plt.show()


if 'pipeline' in options:
    # Define HOG parameters
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2

    #Define training image paths
    car_train_path = r'C:\Users\mes59\Documents\Udacity\SDC\Term 1\Project 5\Training Data\vehicles'
    not_car_train_path = r'C:\Users\mes59\Documents\Udacity\SDC\Term 1\Project 5\Training Data\non-vehicles'

    #Extract training features. Currently just hog features.
    print("Extracting features from training data...")
    t = time.time()
    car_features = get_training_features(car_train_path, orient, pix_per_cell, cell_per_block)
    not_car_features = get_training_features(not_car_train_path, orient, pix_per_cell, cell_per_block)
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
    test_imgs = os.listdir(test_imgs_path)

    for i in range(len(test_imgs)):
        img_path = test_imgs_path + '\\' + test_imgs[i]
        img = mpimg.imread(img_path)
        small_windows = slide_window(img, x_start_stop=[None, None], y_start_stop=[int(img.shape[0]/2), 500],
                        xy_window=(64,64), xy_overlap=(0.5, 0.5))
        med_windows = slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                        xy_window=(192,192), xy_overlap=(0.75, 0.75))
        big_windows = slide_window(img, x_start_stop=[None, None], y_start_stop=[0, img.shape[0]],
                        xy_window=(256,256), xy_overlap=(0.75, 0.75))
        windows = [small_windows, med_windows, big_windows]
        hot_windows = search_windows(img, svc, windows, X_scaler, orient, pix_per_cell, cell_per_block)
        window_img = draw_boxes(img, hot_windows)
        plt.imshow(window_img)
        plt.show()
