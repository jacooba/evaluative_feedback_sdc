import os

# For reference, size of original images:
# IN_WIDTH = 320
# IN_HEIGHT = 160
# IN_CHANNELS = 3

#loss for policy net (only comuted within pnet, and if not using fnet as loss). 
#Note, behavioral clone settings: ALPHA = 0.0, LOSS_EXPONENT=2, THRESHOLD_FEEDBACK = True
ALPHA = 0.5 #[0,1] scale on negative feeback. (knob for making it less important).
LOSS_EXPONENT = 2 #loss = f*abs(y_hat-y)^LOSS_EXPONENT. 2 is mse.
FEEDBACK_IN_EXPONENT = False #if true, loss = abs(y_hat-y)^(f*LOSS_EXPONENT)
THRESHOLD_FEEDBACK = False #if true, loss will be -1(*alpha) or 1
# 3 runs, 80 steps each time, avg angle error, exp=2 ("mixed_2" [100, 300, 20] lr=1e-6):
# exp feedback:                             24.74, 20.66, 17.25 -> avg = 20.88
# scalar feedback:                          3.30,  2.48, 2.61 -> avg = *2.80*

# alpha set to 0.5
# scalar feedback:                          2.96, 3.40, 3.53 -> avg = 3.30

# alphs set to 0 (gradations on pos f)...
# threshold to *clone*:                     4.59, 4.56, 3.02 -> avg = $4.06$
# postive feedback as scalar:               3.33, 2.71, 3.72 -> ave = 3.25
# positive exp feedback:                    3.84, 3.50, 2.80 -> avg = 3.38

# note: angles induced by f-net seem to be more extreme


#note, in our data as of 5/7, the largest magnitude unscalled feedback is currently 21.744000000000025
MAX_ANGLE = 50.0 #the max angle for labels that unity allows
MAX_FEEDBACK_ANGLE = 22.0 #max angle set by us for scaling feedback. (if over, auto f=-0.5).
FEEDBACK_EPS = 5.0 #0.2 #give postive feedback if steer corection within eps of 0.0
#note, all lablled data as of 5/7 was already recorded with an eps of "5"
#except for labeled_data_lanechangeleft, which had no eps so that we could do it later
#it is cutrently set to 5.0 for constistency
#TODO: - recalibrate wheel, make sure resistance is correct, recollect slower (and in correct col)

INPUT_WIDTH = 299
INPUT_HEIGHT = 299
INPUT_DEPTH = 3

INPUT_MEAN = 256.0/2.0
INPUT_STD = 256.0/2.0

#careful how long you make the list. validation can take awhile...
DISCRETE_ANGLES = [angle/MAX_ANGLE for angle in [-15, -10, -8, -6, -5, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 5, 6, 8, 10, 15]] #50 is max angle
#DISCRETE_ANGLES = [angle/MAX_ANGLE for angle in [-7, -2, -1, 0, 1, 2, 7]] #50 is max angle

#TODO 
DROP_RATE = 0.0
BATCH_NORM = False

NUM_EPOCHS = 1 #1
LRATE = 1e-6 #1e-9 
#with f_c = [20, 10] and conv_4: (really should've picked archtirecutre first)
#1e-12 could try but I think I did. likely lower bound.
#1e-9 nice curve .66 to .48 then .40 after 2epc  
#**1e-8**  nice curve 1.49 to .37 then smooth plateu to 0.30 after 2epc
    #2e-8 sharp down to .29 then flat
#1e-7 sharp down to .83 then flat -> too high     
#1e-6 sharp down to .25 then flat -> too high        
#1e-3 flat at 1.7 -> too high
## add more neurons ##
#1e-8 cuases train and val loss to go up
#back to ***1e-9*** which seems to be working... just a bit slower
BATCH_SIZE = 100
EVAL_BATCH_SIZE = 100 #150 #should be as large as possible (needed to fit in mem)

#additonal convolutional layers after inception
CONV_CHANNELS = []
CONV_KERNELS = []
CONV_STRIDES = []

FC_CHANNELS = [100, 300, 20] #[100, 300, 20] #[100, 80, 20] #[20, 10]

#configs tests:
# mixed_4  [100, 80, 20]. great val loss slope, but too slow (straight line from 0.95 to 0.87 took like 6 epochs)
# conv_4 [30, 300, 10]. ^ same  (1.58 to 1.50)
# mixed_4 [20, 10] decreasing stright line super slow (1.34, 1.33)
# *conv_4 [20, 10]. has a curve from (0.32, 0.29) just starting to flatten out
# conv_4 [20, 50, 20]. has stright line  (1.31 to 1.18)
# conv_4 [20, 50]. loss increases (.261, .263)
# conv_4 [20, 200] decreasing slowly from about 1.55 to about 1.47
# conv_4  [100, 80, 20] steep curve (1.70 to 1.58)
# mixed_1  [20, 10]  (.959, .954)
# conv_4 [20, 10, 5] (.997, .92) (pretty stright. slight shallow)
# conv_4 [20, 20, 5] (.84, .76)
# conv_4 [40, 10] (.951, .694) straight. tiiiiny shallow
# conv_4 [40, 20] (.258,.258) increaseing
# conv_4 [20, 200, 10]. (1.2, 1.15) straight
# conv_4 [40, 40] (.60, .41) slight shallow 
# conv_4 [80, 80] (1.69, 1.51) steep 
# ugh they all start at random errors at different trials...
# conv_4 [80, 80] (.54, .31) shallow starting to flat 3 EPOCH (but not enough space to save)
# oh wait, was i training same model w/o new for lAST FEW?? idk .
#tring overnight:
# mixed_2  [100 100] 80 epochs -> crashed

#this string will be appeneded to your summary names
#if you leave tensorboard open and do not change this, 
#it will get confused by all the summaries on the same plot
TRIAL_STR = "" 
#need to change loss and architecture together (or set architecture first)
# "mixed_2_100_300_20_lr=1e-5" was really good! down to  0.1877! (decr lr a bit)
# "mixed_2_100_300_20_lr=1e-6" was really good! down to  0.24! (decr lr a bit)
    #after 3 epochs it got to 1.9999999 but stopped chaning much. really only need 1.5 epc. lower lr???

IMAGE_INPUT_TENSOR_NAME = 'Mul'
BOTTLENECK_TENSOR_NAME = "mixed_2/join:0" #"conv_4/Conv2D:0" #"mixed_10/join:0" #'pool_3/_reshape:0'
#note, there should be 4 convs (1-4), then 10 mixed. The 3 pools are interspersed.
TRAIN_INCEPTION_TOO = True #whether to add a stop gradient after inception layers

SUMMARY_SAVE_FREQ = 80 #55 #50 #100 #30
MODEL_SAVE_FREQ = 9999999999 #200

INCEPTION_DIR = "inception"
INCEPTION_PATH = os.path.join(INCEPTION_DIR, 'classify_image_graph_def.pb')

#policy net paths
P_SUMMARY_DIR = os.path.join(os.getcwd(), "PSummary")
P_MODEL_DIR = os.path.join(os.getcwd(), "PModel")
P_MODEL_PATH = os.path.join(P_MODEL_DIR, "model")
#feedback net paths
F_SUMMARY_DIR = os.path.join(os.getcwd(), "FSummary")
F_MODEL_DIR = os.path.join(os.getcwd(), "FModel")
F_MODEL_PATH = os.path.join(F_MODEL_DIR, "model")

EVERY_X_IMAGES = 5 #down sample the frequency of images (for processing)
VAL_FRAC = .25

TRAIN_DATA_PATH = "trainData" # data
VAL_DATA_PATH = "valData"

TRAIN_LABELS_PATH = "trainLabels" # labels
VAL_LABELS_PATH = "valLabels"

TRAIN_FEEDBACK_PATH = "trainFeedback" # evaluative feedback
VAL_FEEDBACK_PATH = "valFeedback"

