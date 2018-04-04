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




def getIMGPathAndLabelTups(csvFilename, dirpath):

    csvFilepath = os.path.join(dirpath, csvFilename)
    with open(csvFilepath, "r") as csvFile:
        fileLines = csvFile.readlines()

    toReturn = []

    for i, line in enumerate(fileLines):
        if i%c.EVERY_X_IMAGES==0: #only take every "x" image. we have too much data
            lineItems = line.split(",")
            windowsPath, label, feedback = lineItems[0], float(lineItems[3]), lineItems[4] #float(lineItems[4])
            imageName = windowsPath.split("IMG\\")[1]
            path = os.path.join(dirpath, "IMG")
            path = os.path.join(path, imageName)

            print "IF FEEDBACK IS NULL, setting to 999 for testing purposes... PLEASE REMOVE AND CONVERT TO FLOAT above ONCE DONE COLLECTING FEEDBACK"
            if feedback == "null":
                feedback = 999
            else:
                feedback = float(feedback)

            toReturn.append( (path,label,feedback) )

    return toReturn

#NOTE: saves as (h, w, channels)
def getIMGFromPath(path):
    img = Image.open(path)
    return np.array(img)




if __name__ == "__main__":

    imagepathLabelFeedbackTups = [] #list of tuple of image paths and steering angle labels
    for dirpath, dirnames, filenames in os.walk(os.getcwd()):
        for csvFilename in fnmatch.filter(filenames, '*.csv'):
            imagepathLabelFeedbackTups.extend(getIMGPathAndLabelTups(csvFilename, dirpath))
    random.shuffle(imagepathLabelFeedbackTups)  #shuffle the data

    #calculate number of images for val/train
    numData = len(imagepathLabelFeedbackTups)
    numDataVal = int(c.VAL_FRAC*numData)
    print "\nnumber of images before 2x augment:", numImgs,"\n"

    #seperate data
    valTups = imagepathLabelFeedbackTups[:numImgVal]
    trainTups = imagepathLabelFeedbackTups[numImgVal:]

    #unzip into lists of serpeate compenents
    trainImagePaths, trainLables, trainFeedback = zip(*trainTups) #extract list of images and list of lables and list of feedback
    valImagePaths, valLables, valFeedback = zip(*valTups)

    #convert paths to actual image data
    print "Reading in image data..."
    t = time()
    valData = [getIMGFromPath(path) for path in valPathData]
    trainData = [getIMGFromPath(path) for path in trainPathData]
    print "Took", (time()-t)/60.0, "minutes\n"

    #augment data by flipping image and inverting angle
    print "augmenting data..."
    t = time()
    trainData.extend([np.fliplr(img) for img in trainData]) #train data
    trainLables.extend([-1.0*label for label in trainLables])
    trainFeedback.extend(trainFeedback)
    valData.extend([np.fliplr(img) for img in valData]) #val data
    valLables.extend([-1.0*label for label in valLables])
    valFeedback.extend(valFeedback)
    print "Took", (time()-t)/60.0, "minutes\n"

    #save data as numpy arrays
    print "Writing out data..."
    t = time()
    np.save(c.VAL_LABELS_PATH, valLabels)
    np.save(c.TRAIN_LABELS_PATH, trainLabels)
    np.save(c.VAL_DATA_PATH, valData)
    np.save(c.TRAIN_DATA_PATH, trainData)
    np.save(c.VAL_FEEDBACK_PATH, valFeedback)
    np.save(c.TRAIN_FEEDBACK_PATH, trainFeedback)
    print "Took", (time()-t)/60.0, "minutes\n"

    #NOTE: when you go to load, do np.load("xxxxx.npy"). Dont forget the ".npy"!


