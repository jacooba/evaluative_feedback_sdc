"""
This file takes every xth center image from data folder, shuffles them,
then saves a numpy array file representing all pics and a numpy file
representing all labels (for both train and val data).

We may also later do some pre-processing here of the image, such as normalizing
the input.

NOTE:
In the csv, the data is stored as:
    centerPath, leftPath, rightPath, sample.steeringAngle, sample.throttle,
    sample.brake, sample.speed, sample.position.x, sample.position.y,
    sample.position.z, sample.rotation.w, sample.rotation.x, sample.rotation.y,
    sample.rotation.z
"""

import constants as c
import os
import fnmatch
import random
import numpy as np
from PIL import Image
from time import time


def get_img_path_and_label_tups(csv_filename, dirpath):
    """ Returns a list of tuples of (path, label) each corresponding to an image.

        :param csv_filename: The filename of the CSV containing the data
        :param dirpath: The path to the directory containing the CSV

        :returns: A list of tuples of (path, label) each corresponding to an image.
    """
    csv_filepath = os.path.join(dirpath, csv_filename)
    with open(csv_filepath, "r") as csv_file:
        file_lines = csv_file.readlines()

    image_data = []
    for i, line in enumerate(file_lines):
        if i % c.EVERY_X_IMAGES == 0: #only take every "x" image. we have too much data
            line_items = line.split(",")
            windows_path, label = line_items[0], float(line_items[3])
            image_name = windows_path.split("IMG\\")[1]
            path = os.path.join(dirpath, "IMG", image_name)
            image_data.append((path,label))

    return image_data

#NOTE: saves as (h, w, channels)
def get_img_from_path(path):
    """ Returns the given image as a numpy array.

        :param path: The path to the image

        :returns: The image as a numpy array
    """
    img = Image.open(path)
    return np.array(img)


if __name__ == "__main__":

    image_path_label_tups = [] #list of tuple of image paths and steering angle labels
    for dirpath, dirnames, filenames in os.walk(os.getcwd()):
        for csv_filename in fnmatch.filter(filenames, '*.csv'):
            image_path_label_tups.extend(get_img_path_and_label_tups(csv_filename, dirpath))
    random.shuffle(image_path_label_tups)  #shuffle the data
    image_paths_in_new_order, lables_in_new_order = zip(*image_path_label_tups) #extract list of images and lables

    #calculate number of images for val/train
    num_imgs = len(image_paths_in_new_order)
    num_img_val = int(c.VAL_FRAC*num_imgs)
    print "\nnumber of images:", num_imgs,"\n"

    #seperate data
    val_path_data = image_paths_in_new_order[:num_img_val] #cut of first for validation
    train_path_data = image_paths_in_new_order[num_img_val:] #cut of first for validation

    val_labels = lables_in_new_order[:num_img_val]
    train_labels = lables_in_new_order[num_img_val:]
    # TODO: read this in as well instead of setting to 1!
    feedback = np.ones_like(train_labels)

    #convert paths to actual image data
    print "Reading in data..."
    t = time()
    val_data = [get_img_from_path(path) for path in val_path_data]
    train_data = [get_img_from_path(path) for path in train_path_data]
    print "Took", (time()-t)/60.0, "minutes\n"

    #save data as numpy arrays
    print "Writing out data..."
    t = time()
    np.save(c.VAL_LABELS_PATH, val_labels)
    np.save(c.TRAIN_LABELS_PATH, train_labels)
    np.save(c.VAL_DATA_PATH, val_data)
    np.save(c.TRAIN_DATA_PATH, train_data)
    np.save(c.TRAIN_FEEDBACK_PATH, feedback)
    print "Took", (time()-t)/60.0, "minutes\n"

    #NOTE: when you go to load, do np.load("xxxxx.npy"). Dont forget the ".npy"!
