#This fle takes all center images from data folder, shuffles them, 
#then saves a numpy array file representing all pics and a numpy file representing all labels
#(for both train and val data)

#it may also do some pre-processing of the image, such as normalizing the input



#NOTE:
#in the csv, as of making this, the data is stored as...
# {0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13}\n", 
# = centerPath, leftPath, rightPath, sample.steeringAngle,
#   sample.throttle, sample.brake, sample.speed, sample.position.x, sample.position.y, sample.position.z, 
#   sample.rotation.w, sample.rotation.x, sample.rotation.y, sample.rotation.z

import os
import fnmatch
import random

import numpy as np

from PIL import Image
from time import time

import constants as c




def getIMGPathAndLabelTups(csvFilename, dirpath):

    csvFilepath = os.path.join(dirpath, csvFilename)
    with open(csvFilepath, "r") as csvFile:
        fileLines = csvFile.readlines()

    toReturn = []

    for i, line in enumerate(fileLines):
        if i%c.EVERY_X_IMAGES==0: #only take every "x" image. we have too much data
            lineItems = line.split(",")
            windowsPath, label = lineItems[0], float(lineItems[3])
            imageName = windowsPath.split("IMG\\")[1]
            path = os.path.join(dirpath, "IMG")
            path = os.path.join(path, imageName)

            toReturn.append( (path,label) )

    return toReturn

#NOTE: saves as (h, w, channels)
def getIMGFromPath(path):
    img = Image.open(path)
    return np.array(img)




if __name__ == "__main__":

    imagePathLabelTups = [] #list of tuple of image paths and steering angle labels
    for dirpath, dirnames, filenames in os.walk(os.getcwd()):
        for csvFilename in fnmatch.filter(filenames, '*.csv'):
            imagePathLabelTups.extend(getIMGPathAndLabelTups(csvFilename, dirpath))
    random.shuffle(imagePathLabelTups)  #shuffle the data
    imagePathsInNewOrder, lablesInNewOrder = zip(*imagePathLabelTups) #extract list of images and lables

    #calculate number of images for val/train
    numImgs = len(imagePathsInNewOrder)
    numImgVal = int(c.VAL_FRAC*numImgs)
    print "\nnumber of images:", numImgs,"\n"

    #seperate data
    valPathData = imagePathsInNewOrder[:numImgVal] #cut of first for validation
    trainPathData = imagePathsInNewOrder[numImgVal:] #cut of first for validation

    valLabels = lablesInNewOrder[:numImgVal]
    trainLabels = lablesInNewOrder[numImgVal:]
    # TODO: read this in as well instead of setting to 1!
    feedback = np.ones_like(trainLabels)

    #convert paths to actual image data
    print "Reading in data..."
    t = time()
    valData = [getIMGFromPath(path) for path in valPathData]
    trainData = [getIMGFromPath(path) for path in trainPathData]
    print "Took", (time()-t)/60.0, "minutes\n"

    #save data as numpy arrays
    print "Writing out data..."
    t = time()
    np.save(c.VAL_LABELS_PATH, valLabels)
    np.save(c.TRAIN_LABELS_PATH, trainLabels)
    np.save(c.VAL_DATA_PATH, valData)
    np.save(c.TRAIN_DATA_PATH, trainData)
    np.save(c.TRAIN_FEEDBACK_PATH, feedback)
    print "Took", (time()-t)/60.0, "minutes\n"

    #NOTE: when you go to load, do np.load("xxxxx.npy"). Dont forget the ".npy"!


