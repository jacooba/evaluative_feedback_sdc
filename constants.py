import os

# For reference, size of original images:
# IN_WIDTH = 320
# IN_HEIGHT = 160
# IN_CHANNELS = 3

#loss for policy net (only comuted within pnet, and if not using fnet as loss). 
#Note, behavioral clone settings: ALPHA = 0.0, LOSS_EXPONENT=2, THRESHOLD_FEEDBACK = True
ALPHA = 1.0 #1e-10 #1.0 #[0,1] scale on negative feeback. (knob for making it less important. deoends on data collection. 2x neg data may reuire 0.5x).
LOSS_EXPONENT = 2 #1 #2 #loss = f*abs(y_hat-y)^LOSS_EXPONENT. 2 is mse.
FEEDBACK_IN_EXPONENT = False #if true, loss = abs(y_hat-y)^(f*LOSS_EXPONENT)
THRESHOLD_FEEDBACK = False #if true, loss will be -1(*alpha) or 1
SIGN_IN_EXPONENT = False #if true, sign of f will move to exp. loss = |f|(y_hat-y)^(2*sign(f))
assert not (FEEDBACK_IN_EXPONENT and SIGN_IN_EXPONENT) #only do one of these
# note -- these losses were just calculated as difference in angle on pos data. 
# but really not entirely appropriate since the loss favors cloning
# really, it should get less loss for turning more in the right direction (not more since its farther from label)
# weighting by feedback might help but still not quite correct
# e.g. gradations in pos data were told to care about data unevenly, but tested evenly
# also we are hoping feedback helps so it doesnt learn to do eaxctly what we did, but better than what we did
# --val split here was 75-25
# 3 runs, 80 steps each time, avg angle error, exp=2 ("mixed_2" [100, 300, 20] lr=1e-6):
# exp feedback:                             16.88, 34.35, 16.93 -> avg = 
# scalar feedback:                          2.42, 2.42, 3.06 -> avg = *2.63*

# alpha set to 0.5
# scalar feedback:                           2.75, 2.57, 3.14 -> avg = 
# exp feedback:                              9.68 -> avg =                          

# alphs set to 0 (gradations on pos f)...
# threshold to *clone*:                      2.52, 2.37, 2.59 -> avg = $2.49$
# postive feedback as scalar:                3.56, 3.41, 3.96 -> avg = 
# positive exp feedback:                     -> avg =  

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

NUM_EPOCHS = 5 #1
LRATE = 1e-6 #1e-6 #1e-9 
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
# with [100, 300, 20], f-net seems to vary a lot with image, but not with angle.
# likely enough to predict by which way you are facing then which way you steer
# e.g. if heading off road, those will all have same img, but only split second with different label
# before change occurs
# really, that state where you start to turn back is the only one where it is pointed off road but has seen both
# angles. In this case, -15 or 15 probably is best.
#### f-preds by angle and image
# [[[0.27463108 0.8966497  0.8142399  ... 0.61680347 0.70496565 0.85056734]]
#  [[0.27106214 0.8953991  0.8142741  ... 0.6152304  0.70133996 0.8501611 ]]
#  [[0.26963258 0.8948949  0.81419414 ... 0.6144671  0.6998794  0.8499993 ]]
#  ...
#  [[0.25777608 0.8907649  0.8126278  ... 0.6096654  0.68775004 0.84868026]]
#  [[0.25635573 0.89021856 0.81237054 ... 0.60916096 0.68617576 0.84817505]]
#  [[0.2523774  0.8888414  0.8117536  ... 0.607898   0.68272495 0.8468019 ]]]
####
# also tried with [10,5]... not better. pretty much all 15.

#configs tests (tuning records):
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
#tuning records... need to change loss and architecture together (or set architecture first)
#f_net...
# "mixed_2_100_300_20_lr=1e-5" was really good! down to  0.1877! (decr lr a bit)
# $$"mixed_2_100_300_20_lr=1e-6" was really good! down to  0.24! (decr lr a bit)
    #after 3 epochs it got to 1.9999999 but stopped chaning much. really only need 1.5 epc. higher lr???
#p-net.. 
# "mixed_2_100_300_20_lr=1e-5_pnet" -> get down to 2 but looks like it starts to overfit. but may be an artificat of error metric 
# $$"mixed_2_100_300_20_lr=1e-6_pnet" -> sharp down, but then still linearly going down to ~2.7 after 2 epc
# "mixed_2_100_300_20_lr=1e-7_pnet" -> flattens out after two epcs to 3.27 (not good enough)
#p-net exp..
# "mixed_2_100_300_20_lr=1e-6_pnet_exp" -> in terms of error, there are some bumps but curve looks good after ~10 epcs
                                           #lower or higher lr didnt work well i think

#### FULL MODELS SAVED (will be on my google drive) ####
# CLONE #
# "mixed_2_100_300_20_lr=1e-6_pnet_clone" clone 5 epc -> err = 1.915
# "_clone_trial_2___valerr=2.014_loss=0.0035"
# "_clone_trial_3___valerr=1.874_loss=0.0031"
# "_clone_5xlr___valerr=2.027_loss=0.0033"
# "_clone_10xlr___valerr=1.681_loss=0.0026"
# "_clone_15xlr___valerr=2.232_loss=0.0039" 
# "_clone_20xlr___valerr=2.355_loss=0.0042" 
# "_clone_29xlr___valerr=_loss=" #diverged, not saved (first lr diverge within 150 steps)


# SCALAR #
# "mixed_2_100_300_20_lr=1e-6_pnet" f-scalar 5 epc -> err = 1.994
# "_scalar_trial_2___valerr=2.374_loss=0.0020" *loss still decreasing*
# "_scalar_2xlr___valerr=2.555_loss=0.000083"
# "_scalar_5xlr___valerr=3.130_loss=-.0016"
# "_scalar_10xlr___valerr=2.057_loss=0.0011"
# "_scalar_THRESHOLD_alpha_1.0___valerr=2.648_loss=0.0027"
# "_scalar_THRESHOLD_alpha_0.5___valerr=2.012_loss=0.0031"
# "_scalar_THRESHOLD_alpha_0.1___valerr=1.843_loss=0.0030"
# "_scalar_alpha_0.5___valerr= 2.261_loss=0.0033"
# "_scalar_alpha_0.5_5xlr___valerr=2.337_loss=0.0021"
# "_scalar_alpha_0.5_5xlr_THRESHOLD___valerr=1.940_loss=0.0025"
# "_scalar_alpha_0.25___valerr=2.352_loss=0.0036"
# "_scalar_alpha_0.1___valerr=2.063_loss=0.0030"
# "_scalar_alpha_0___valerr=2.367_loss=0.0038"
# "_scalar_alpha_0_10xlr___valerr=1.751_loss=0.0024"
# "_scalar_alpha_0_5xlr___valerr=1.880_loss=0.0025"
# ..redoing best one in terms of performance..
# "_trial_2_scalar_10xlr___valerr=3.104_loss=-0.0035"
# "_trial_3_scalar_10xlr___valerr=3.016_loss=-0.0040"
# "_scalar_pow1___valerr=1.857_loss=1.405"
# "_scalar_pow1_10xlr___valerr=2.021_loss=1.428"
# "_scalar_pow1_5xlr___valerr=1.949_loss=.02638"
# "_scalar_15xlr___valerr=2.938_loss=-.0041"
# "_scalar_20xlr___valerr=45.57_loss=0.75" #diverged, so redone
# "_&scalar_20xlr___valerr=3.411_loss=-0.0027"
# "_scalar_46xlr___valerr=_loss=" #diverged, not saved (had been first lr diverge within 150 steps)


# invert # (sign in exp)
# "_invert___valerr=14.94_loss=1.389"
# "_invert_5xlr_1e-10alpha___valerr=2.949_loss=5.106e-3" (loss started incr. overfit?)
# "_invert_10xlr_1e-8alpha___valerr=6.976_loss=0.03103" (should probs decr lr actually... later)
# "_invert_10xlr_1e-9alpha___valerr=3.925_loss=0.01018"
# "_invert_10xlr_1e-10alpha___valerr=2.572_loss=4.254e-3"
# "_invert_1e-10alpha___valerr=2.615_loss=4.270e-3" ** (curve more normal than above tho)
# "_invert_1e-9alpha___valerr=5.38_loss=.01631"
# "_invert_1e-8alpha___valerr=4.236_loss=.01090"
# "_invert_1e-7alpha___valerr=4.053_loss=.01014"
# "_invert_1e-6alpha___valerr=28.24_loss=.3194"

# exp #
# "mixed_2_100_300_20_lr=1e-6_pnet_exp" f-exp 11 epc -> err = 11.24, loss=0.3817
# "mixed_2_100_300_20_lr=1e-6_pnet_exp_alphahalf" f-exp a=0.5  -> err = seems to diverge model not saved
# "mixed_2_100_300_20_lr=1e-6_pnet_exp_alphatenth" f-exp a=0.5 2 epc -> err = 5.84 5 epc-> 4.978, loss=0.1544
#^couldnt get err even close to 2. thresholding might help this.
# "_exp_THRESHOLD___valerr=32.316_loss=32.32" ->diverged. model not saved.
# "_exp___valerr=29.03_loss=32.25" ->diverged. not saved.
# "_exp_THRESHOLD_alpha_0.1___valerr=5.876_loss=0.1699"

# f-net #
# "mixed_2_100_300_20_lr=1e-6_fnet" f-net 5 epc -> (f)err = 0.1988
####

#note on f: we didnt add to angle and do behave clone on that b/c turns are do hard
# to get right if you cant see outcome of your action. only realtive works.

IMAGE_INPUT_TENSOR_NAME = 'Mul'
BOTTLENECK_TENSOR_NAME = "mixed_2/join:0" #"conv_4/Conv2D:0" #"mixed_10/join:0" #'pool_3/_reshape:0'
#note, there should be 4 convs (1-4), then 10 mixed. The 3 pools are interspersed.
TRAIN_INCEPTION_TOO = True #whether to add a stop gradient after inception layers

SUMMARY_SAVE_FREQ = 50 #80 #55 #50 #100 #30
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
VAL_FRAC = .15
#right now, using 17,918 train and 3,162 val data after 2x aug

TRAIN_DATA_PATH = "trainData" # data
VAL_DATA_PATH = "valData"

TRAIN_LABELS_PATH = "trainLabels" # labels
VAL_LABELS_PATH = "valLabels"

TRAIN_FEEDBACK_PATH = "trainFeedback" # evaluative feedback
VAL_FEEDBACK_PATH = "valFeedback"

