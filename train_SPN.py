"""
SPN train module
@authorized by Shasha Bae
@description: train the SPN model
"""

import random
import argparse
from os.path import join
from glob import glob

import tensorflow as tf
from tensorflow.python import keras
from sklearn.model_selection import train_test_split

from utils import *
from network import *




def main():

	################################################## parse the arguments
	parser = argparse.ArgumentParser(description='Train retouch detection network.')
	parser.add_argument('--src_path', type=str, 			default='E:\\paired_minibatch', help='source path')
	parser.add_argument('--train_path', type=str, 			default='./train_*.txt', help='source path')
	parser.add_argument('--test_path', type=str, 			default='./test_*.txt', help='source path')
	parser.add_argument('--validation_path', type=str, 		default='./validation_*.txt', help='source path')

	parser.add_argument('--method', type=str, 				default="multi", help='blur, median, noise or multi')
	parser.add_argument('--br', type=str, 					default="*", help='bitrate')
	parser.add_argument('--log_path', type=str, 			default='./logs', help='log path')
	parser.add_argument('--summary_interval', type=int, 	default=1, help='loss inverval')
	parser.add_argument('--checkpoint_interval', type=int, 	default=1, help='checkpoint interval')

	parser.add_argument('--network', type=str, 				default="DCTNet", help='SRNet or MISLNet or NamNet or MMCNet or DCTNet or MesoNet')
	parser.add_argument('--network_scale', type=float, 		default=1.0, help='network scale')
	parser.add_argument('--checkpoint_path', type=str, 		default="", help='checkpoint path')
	parser.add_argument('--regularizer', type=float, 		default=0.0001, help='regularizer')
	
	parser.add_argument('--epoch', type=int, 				default=30, help='epoch')
	parser.add_argument('--batch_size', type=int, 			default=32, help='batch size')
	parser.add_argument('--start_lr', type=float, 			default=1e-04, help='start learning rate')
	parser.add_argument('--lr_update_interval', type=int, 	default=1, help='learning rate update interval')
	parser.add_argument('--lr_update_rate', type=float, 	default=0.9, help='learning rate update rate')

	parser.add_argument('--debug', type=bool, 				default=False, help='True or False')
	
	args = parser.parse_args()

	SRC_PATH 			= args.src_path
	TRAIN_PATH 			= args.train_path
	TEST_PATH 			= args.test_path
	VALIDATION_PATH 	= args.validation_path

	METHOD 				= args.method
	BITRATE 			= args.br + "k"
	LOG_PATH 			= args.log_path
	SUMMARY_INTERVAL 	= args.summary_interval
	CHECKPOINT_INTERVAL = args.checkpoint_interval

	NETWORK 			= args.network
	SCALE 				= args.network_scale
	CHECKPOINT_PATH 	= args.checkpoint_path
	REG 				= args.regularizer
	
	EPOCHS 				= args.epoch
	BATCH_SIZE 			= args.batch_size
	START_LR 			= args.start_lr
	LR_UPDATE_INTERVAL 	= args.lr_update_interval
	LR_UPDATE_RATE 		= args.lr_update_rate

	DEBUG 				= args.debug
	
	print_args(args)



	################################################## Setup the dataset
	# Set train, validation, and test data
	train_fnames = txt2list(glob(TRAIN_PATH))
	train_fnames = train_fnames[:int(len(train_fnames) * 8 / 9)]
	test_fnames = txt2list(glob(TEST_PATH))
	valid_fnames = txt2list(glob(VALIDATION_PATH))
	


	# Load data
	valid_dataset  	= configure_dataset(valid_fnames, BATCH_SIZE)
	train_dataset 	= configure_dataset(train_fnames, BATCH_SIZE)
	test_dataset 	= configure_dataset(test_fnames, BATCH_SIZE)
	


	################################################## Setup the training options
	# Load model
	NUM_CLASS = 4 if METHOD == "multi" else 2
	
	model = load_model(NETWORK, SCALE, REG, NUM_CLASS)
	load_ckpt(model, CHECKPOINT_PATH, START_LR)



	################################################## Train the model
	# Set callback functions 
	LOG_PATH, callbacks = load_callbacks(args)


	# write the argument information
	write_args(args, join(LOG_PATH, "setup.txt"))


	# Set number of steps per epoch
	STEPS_PER_EPOCH_TRAIN = 2 * len(train_fnames) // BATCH_SIZE
	STEPS_PER_EPOCH_VALID = 2 * len(valid_fnames) // BATCH_SIZE


	# Train the model
	history = model.fit(train_dataset, \
						epochs=EPOCHS, \
						steps_per_epoch=STEPS_PER_EPOCH_TRAIN, \
						callbacks=callbacks, \
						validation_data=valid_dataset, \
						validation_steps=STEPS_PER_EPOCH_VALID, \
						verbose=1)

	write_history(history, join(LOG_PATH, "train.txt"))



	################################################## Test the model
	STEPS_TEST = 2 * len(test_fnames) // BATCH_SIZE

	result = model.evaluate(test_dataset, steps=STEPS_TEST)
	
	write_result(["Loss", "Accuracy"], result, join(LOG_PATH, "test.txt"))










if __name__=="__main__":
	main()


