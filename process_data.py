#This fle takes all center images from data folder, shuffles them,
#then saves a numpy array file representing all pics and a numpy file representing all labels
#(for both train and val data)

#it may also do some pre-processing of the image, such as normalizing the input



#NOTE:
#in the csv, as of making this, the data is stored as...
# centerPath, leftPath, rightPath, sample.steeringAngle, feedback,
# sample.throttle, sample.brake, sample.speed, sample.position.x, sample.position.y, sample.position.z,
# sample.rotation.w, sample.rotation.x, sample.rotation.y, sample.rotation.z

import os
import fnmatch
import random

import numpy as np

from PIL import Image
from time import time

import constants as c




def get_img_path_and_label_tups(csv_filename, dirpath):
    csv_filepath = os.path.join(dirpath, csv_filename)
    with open(csv_filepath, "r") as csv_file:
        file_lines = csv_file.readlines()

    to_return = []

    for i, line in enumerate(file_lines):
        if i % c.EVERY_X_IMAGES == 0: #only take every "x" image. we have too much data
            line_items = line.split(",")
            windows_path, label, unprocessed_feedback = line_items[0], float(line_items[3]), float(line_items[4]) #float(line_items[4])
            feedback = process_angle_and_corection_to_feedback(label, unprocessed_feedback)
            image_name = windows_path.split("IMG\\")[1]
            path = os.path.join(dirpath, "IMG")
            path = os.path.join(path, image_name)

            # print("IF FEEDBACK IS NULL, setting to 999 for testing purposes... PLEASE REMOVE AND CONVERT TO FLOAT above ONCE DONE COLLECTING FEEDBACK")
            # if feedback == "null":
            #     feedback = 999
            # else:
            #     feedback = float(feedback)

            to_return.append( (path,label,feedback) )

    return to_return

#NOTE: saves as (h, w, channels)
def get_img_from_path(path):
    img = Image.open(path)
    return np.array(img)

def process_angle_and_corection_to_feedback(theta, cor):
    # if angle is greater than 75 degrees, its so bad it should be negative
    if abs(cor) >= c.MAX_FEEDBACK_ANGLE:
        return -0.5
    
    #scale
    cor_scaled = cor/c.MAX_FEEDBACK_ANGLE

    #see if good or bad
    if ((theta>0) == (cor>0)) or (abs(cor)<=c.FEEDBACK_EPS): #signs are equal or theta within some eps of 0
        f = 1.0 - abs(cor_scaled)
    else:
        f = -abs(cor_scaled)

    return f



if __name__ == "__main__":

    image_path_label_feedback_tups = [] #list of tuple of image paths and steering angle labels
    for dirpath, dirnames, filenames in os.walk(os.path.join(os.getcwd(), "eval_feedback_data")):
        for csv_filename in fnmatch.filter(filenames, '*.csv'):
            image_path_label_feedback_tups.extend(get_img_path_and_label_tups(csv_filename, dirpath))
    random.shuffle(image_path_label_feedback_tups)  #shuffle the data

    #calculate number of images for val/train
    num_data = len(image_path_label_feedback_tups)
    num_data_val = int(c.VAL_FRAC * num_data)
    print("\nnumber of images before 2x augment:", num_data,"\n")

    #seperate data
    val_tups = image_path_label_feedback_tups[:num_data_val]
    train_tups = image_path_label_feedback_tups[num_data_val:]

    #unzip into lists of serpeate compenents
    train_image_paths, train_labels, train_feedback = zip(*train_tups) #extract list of images and list of labels and list of feedback
    train_labels, train_feedback = list(train_labels), list(train_feedback)
    val_image_paths, val_labels, val_feedback = zip(*val_tups)
    val_labels, val_feedback = list(val_labels), list(val_feedback)
    # print(max(max(val_feedback), max(train_feedback)))
    # # the largest magnitude (unsclaed) is currently 21.744000000000025 as of 5/7
    # exit()

    #convert paths to actual image data
    print("Reading in image data...")
    t = time()
    val_data = [get_img_from_path(path) for path in val_image_paths]
    train_data = [get_img_from_path(path) for path in train_image_paths]
    print("Took", (time()-t)/60.0, "minutes\n")

    #augment data by flipping image and inverting angle
    print("augmenting data...")
    t = time()
    train_data.extend([np.fliplr(img) for img in train_data]) #train data
    train_labels.extend([-1.0*label for label in train_labels])
    train_feedback.extend(train_feedback)
    val_data.extend([np.fliplr(img) for img in val_data]) #val data
    val_labels.extend([-1.0*label for label in val_labels])
    val_feedback.extend(val_feedback)
    print("Took", (time()-t)/60.0, "minutes\n")

    #save data as numpy arrays
    print("Writing out data...")
    t = time()
    np.save(c.VAL_LABELS_PATH, val_labels)
    np.save(c.TRAIN_LABELS_PATH, train_labels)
    np.save(c.VAL_DATA_PATH, val_data)
    np.save(c.TRAIN_DATA_PATH, train_data)
    np.save(c.VAL_FEEDBACK_PATH, val_feedback)
    np.save(c.TRAIN_FEEDBACK_PATH, train_feedback)
    print("Took", (time()-t)/60.0, "minutes\n")

    #NOTE: when you go to load, do np.load("xxxxx.npy"). Dont forget the ".npy"!
