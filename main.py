import time
import sys
import numpy as np
import cv2
import os
import glob
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from scipy.ndimage.measurements import label
from sklearn.svm import LinearSVC
from imageProcessing import *
from slidingWindow import *
from heatMap import *

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
        small_windows = slide_window(img, x_start_stop=[None, None], y_start_stop=[380, 500],
                        xy_window=(64,64), xy_overlap=(.75,.75))
        med_windows = slide_window(img, x_start_stop=[None, None], y_start_stop=[None, 550],
                        xy_window=(160,160), xy_overlap=(.75,.75))
        big_windows = slide_window(img, x_start_stop=[None, None], y_start_stop=[336,img.shape[0]],
                        xy_window=(256,256), xy_overlap=(.5,.5))
        img = draw_boxes(img, big_windows, color=(255, 0, 0), thick=6)
        img = draw_boxes(img, med_windows, color=(0, 255, 0), thick=3)
        img = draw_boxes(img, small_windows, color=(0, 0, 255), thick=1)

        plt.imshow(img)
        plt.show()

def train_classifier(orient, pix_per_cell, cell_per_block, color_space, spatial_size, hist_bins, hog_channel, spatial_feat, hist_feat, hog_feat, save_model):
    #Get training image paths
    car_train_path = r'C:\Users\mes59\Documents\Udacity\SDC\Term 1\Project 5\Training Data\vehicles'
    not_car_train_path = r'C:\Users\mes59\Documents\Udacity\SDC\Term 1\Project 5\Training Data\non-vehicles'
    vehicle_files = [f for f in glob.glob(car_train_path + '/**/*.png', recursive=False)]
    non_vehicle_files = [f for f in glob.glob(not_car_train_path + '/**/*.png', recursive=False)]

    if save_model:
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

        joblib.dump(svc, 'svc.pkl') #save classifier
        print("Saved classifier!")
        joblib.dump(X_scaler, 'X_scaler.pkl') #save X_scaler
        print("Saved X_scaler!")

        return svc, X_scaler, X_train, X_test, y_train, y_test

    else:
        print("Loading saved classifier...")
        svc = joblib.load("svc.pkl")
        X_scaler = joblib.load("X_scaler.pkl")
        print("Saved classifier loaded!")
        return svc, X_scaler, None, None, None, None

if 'pipeline_images' in options:
    # Define festure extraction parameters
    orient = 9 #HOG
    pix_per_cell = 8 #HOG
    cell_per_block = 2 #HOG
    hog_channel = 'ALL' #HOG
    color_space = 'RGB'
    spatial_size = (16,16) #spatial binning
    hist_bins = 16 #color channel histograms
    spatial_feat = True
    hist_feat = True
    hog_feat = True

    svc, X_scaler, X_train, X_test, y_train, y_test = train_classifier(orient, pix_per_cell, cell_per_block, color_space, spatial_size, hist_bins, hog_channel, spatial_feat, hist_feat, hog_feat)

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

if 'pipeline_video' in options:
    # Define festure extraction parameters
    orient = 9 #HOG
    pix_per_cell = 8 #HOG
    cell_per_block = 2 #HOG
    hog_channel = 'ALL' #HOG
    color_space = 'YCrCb'
    spatial_size = (32,32) #spatial binning
    hist_bins = 32 #color channel histograms
    spatial_feat = True
    hist_feat = True
    hog_feat = True

    save_model = True #Do I need to train a classifier? False if classifier is already saved
    svc, X_scaler, X_train, X_test, y_train, y_test = train_classifier(orient, pix_per_cell, cell_per_block, color_space, spatial_size, hist_bins, hog_channel, spatial_feat, hist_feat, hog_feat, save_model)

    if save_model:
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    video_path = r'C:\Users\mes59\Documents\Udacity\SDC\Term 1\Project 5\CarND-Vehicle-Detection-master\project_video.mp4'
    cap = cv2.VideoCapture(video_path)

    #Open video writer
    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # videoout = cv2.VideoWriter('output.avi',fourcc, 30.0,( 1280,720))

    c = 0
    while(cap.isOpened()):
        ret, img = cap.read()
        if not ret:
            break
        c+=1
        if c < 190:
            continue
        # if i % 100 == 0:
        #     print("FRAME: ", i)

        # small_windows = slide_window(img, x_start_stop=[None, None], y_start_stop=[380, 450],
        #                 xy_window=(32,32), xy_overlap=(.5,.5))
        # med_windows = slide_window(img, x_start_stop=[None, None], y_start_stop=[None, 550],
        #                 xy_window=(150,150), xy_overlap=(.75,.75))
        # big_windows = slide_window(img, x_start_stop=[None, None], y_start_stop=[380, img.shape[0]],
        #                 xy_window=(256,256), xy_overlap=(0.9, .8))
        # windows = [small_windows, med_windows, big_windows]
        # hot_windows = search_windows(img, svc, windows, X_scaler, orient, pix_per_cell, cell_per_block, color_space = color_space, spatial_size=spatial_size, hist_bins=hist_bins, hog_channel = hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

        scale = [1.5,1,0.8]
        ystart = [400,400,400]
        ystop = [656,img.shape[0]-160,500]
        hot_windows = []
        for i in range(len(scale)):
            hot_windows = hot_windows + find_cars(img, ystart[i], ystop[i], scale[i], svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        window_img = draw_boxes(img, hot_windows)

        heat = np.zeros_like(img[:,:,0]).astype(np.float)
        # Add heat to each box in box list
        heat = add_heat(heat,hot_windows)
        # Apply threshold to help remove false positives
        heat = apply_threshold(heat,4)
        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)
        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(np.copy(img), labels)

        # #Write to output video
        # videoout.write(draw_img)

        cv2.imshow("Boxed", draw_img)
        cv2.imshow("HeatMap", heatmap)
        cv2.imshow("window_img", window_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
