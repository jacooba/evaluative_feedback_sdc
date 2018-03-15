import constants as c
from retrain_inception import Model
import sys, os
import numpy as np
from utils import get_args, delete_model_files


class Runner:
	def __init__(self):
		""" Initializes a Runner. Reads in data and instantiates a Model. """
		# read in data
		self.train_tup = (None,None,None) #(images,labels,feedbacks)
		self.val_tup = (None,None) #(images,labels)
		self.read_in_data()

		print "\n\nWound up with:"
		print len(self.train_tup[0]), "training images"
		print len(self.val_tup[0]), "validation images\n\n"

		#model
		self.model = Model()

	def read_in_data(self):
		""" Reads in training and validation data and saves them as instance
			variables. """
		print "\nREADING IN DATA..."
		self.train_tup = (np.load(c.TRAIN_DATA_PATH+".npy"),
                          np.load(c.TRAIN_LABELS_PATH+".npy"),
                          np.load(c.TRAIN_FEEDBACK_PATH+".npy"))
		self.val_tup = (np.load(c.VAL_DATA_PATH+".npy"),
                        np.load(c.VAL_LABELS_PATH+".npy"))

	def train(self):
		""" Trains a new model, if there isn't already a saved model file. """
		print "\nTRAINING..."
		self.model.train(self.train_tup, self.val_tup)

	def val(self):
		""" Evaluates the model on the validation data, as long as there is a
			trained model. """
		if not os.path.exists(c.MODEL_PATH):
			print "\n\nNo Model, validating untrained model\n\n"

		print "\nVALIDATING..."
		self.model.eval(self.val_tup)


def main():
	args = get_args()

	if args.new:
		#remove all model files
		delete_model_files()
	if args.train or args.val:
		runner = Runner()

	if args.train:
		runner.train()
	if args.val:
		runner.val()




if __name__ == "__main__":
	main()
