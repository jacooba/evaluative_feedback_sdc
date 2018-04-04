import os

# For reference, size of original images:
# IN_WIDTH = 320
# IN_HEIGHT = 160
# IN_CHANNELS = 3

#loss for policy net (if not using fnet as loss). 
#Note, behavioral clone settings: ALPHA = 0.0, LOSS_EXPONENT=2, THRESHOLD_FEEDBACK = True
ALPHA = 0.8 #[0,1] scale on negative feeback. (knob for making it less important).
LOSS_EXPONENT = 2 #loss = f*abs(y_hat-y)^LOSS_EXPONENT. 2 is mse.
FEEDBACK_IN_EXPONENT = False #if true, loss = abs(y_hat-y)^(f*LOSS_EXPONENT)
THRESHOLD_FEEDBACK = False #if true, loss will be -1(*alpha) or 1

MAX_ANGLE = 50.0

INPUT_WIDTH = 299
INPUT_HEIGHT = 299
INPUT_DEPTH = 3

INPUT_MEAN = 256.0/2.0
INPUT_STD = 256.0/2.0

DISCRETE_ANGLES = [angle/MAX_ANGLE for angle in [-15, -10, -8, -6, -5, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 5, 6, 8, 10, 15]] #50 is max angle

#TODO 
DROP_RATE = 0.0
BATCH_NORM = False

NUM_EPOCHS = 2
LRATE = 0.000001
BATCH_SIZE = 100

CONV_CHANNELS = []
CONV_KERNELS = []
CONV_STRIDES = []

FC_CHANNELS = [20, 10]

IMAGE_INPUT_TENSOR_NAME = 'Mul'
BOTTLENECK_TENSOR_NAME = "conv_4/Conv2D:0" #"mixed_10/join:0" #'pool_3/_reshape:0'

SUMMARY_SAVE_FREQ = 2
MODEL_SAVE_FREQ = 20

INCEPTION_DIR = "inception"
INCEPTION_PATH = os.path.join(INCEPTION_DIR, 'classify_image_graph_def.pb')

#policy net paths
P_SUMMARY_DIR = os.path.join(os.getcwd(), "PSummary")
P_MODEL_DIR = os.path.join(os.getcwd(), "PModel")
P_MODEL_PATH = os.path.join(MODEL_DIR, "model")
#feedback net paths
F_SUMMARY_DIR = os.path.join(os.getcwd(), "PSummary")
F_MODEL_DIR = os.path.join(os.getcwd(), "PModel")
F_MODEL_PATH = os.path.join(MODEL_DIR, "model")

EVERY_X_IMAGES = 5 #down sample the frequency of images
VAL_FRAC = .25

TRAIN_DATA_PATH = "trainData" # data
VAL_DATA_PATH = "valData"

TRAIN_LABELS_PATH = "trainLabels" # labels
VAL_LABELS_PATH = "valLabels"

TRAIN_FEEDBACK_PATH = "trainFeedback" # evaluative feedback
VAL_FEEDBACK_PATH = "valFeedback"

