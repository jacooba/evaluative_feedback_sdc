import os

# For reference, size of original images:
# IN_WIDTH = 320
# IN_HEIGHT = 160
# IN_CHANNELS = 3

INPUT_WIDTH = 299
INPUT_HEIGHT = 299
INPUT_DEPTH = 3

INPUT_MEAN = 256.0/2.0
INPUT_STD = 256.0/2.0

#TODO 
DROP_RATE = 0.0
BATCH_NORM = False

NUM_EPOCHS = 2
LRATE = 0.01
BATCH_SIZE = 50

CONV_CHANNELS = []
CONV_KERNELS = []
CONV_STRIDES = []

FC_CHANNELS = [100]

IMAGE_INPUT_TENSOR_NAME = 'Mul'
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'

SUMMARY_SAVE_FREQ = 50
MODEL_SAVE_FREQ = 1000

INCEPTION_DIR = "inception"
INCEPTION_PATH = os.path.join(INCEPTION_DIR, 'classify_image_graph_def.pb')

SUMMARY_DIR = os.path.join(os.getcwd(), "Summary")
MODEL_DIR = os.path.join(os.getcwd(), "Model")
MODEL_PATH = os.path.join(MODEL_DIR, "model")

EVERY_X_IMAGES = 5 #down sample the frequency of images
VAL_FRAC = .25

TRAIN_DATA_PATH = "trainData" # data
VAL_DATA_PATH = "valData"

TRAIN_LABELS_PATH = "trainLabels" # labels
VAL_LABELS_PATH = "valLabels"

TRAIN_FEEDBACK_PATH = "feedback" # evaluative feedback

