"""
DPN train module
@authorized by Shasha Bae
@description: train the DPN model
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
	parser.add_argument('--src_path', type=str, 			default='./split_names', help='source path')
	parser.add_argument('--src_frac', type=float, 			default=1.0, help='amount of training dataset')

	parser.add_argument('--method', type=str, 				default="multi", help='blur, median, noise or multi')
	parser.add_argument('--br', type=str, 					default="*", help='bitrate')
	parser.add_argument('--log_path', type=str, 			default='./logs', help='log path')
	parser.add_argument('--summary_interval', type=int, 	default=1, help='loss inverval')
	parser.add_argument('--checkpoint_interval', type=int, 	default=1, help='checkpoint interval')

	parser.add_argument('--network1', type=str, 			default="SRNet", help='SRNet or MISLNet or DCTNet or MesoNet')
	parser.add_argument('--checkpoint1_path', type=str, 	default="./logs/20191014_142248_multi_SRNet_93/checkpoint/weights_10", help='checkpoint1 path')

	parser.add_argument('--network2', type=str, 			default="DCTNet", help='SRNet or MISLNet or DCTNet or MesoNet')
	parser.add_argument('--checkpoint2_path', type=str, 	default="./logs/20191016_083522_multi_DCTNet_88/checkpoint/weights_14", help='checkpoint2 path')

	parser.add_argument('--network3', type=str, 			default="Fusion", help='SRNet or MISLNet or DCTNet or MesoNet')
	parser.add_argument('--checkpoint3_path', type=str, 	default="", help='checkpoint3 path')

	parser.add_argument('--regularizer', type=float, 		default=0.0001, help='regularizer')
	
	parser.add_argument('--epoch', type=int, 				default=20, help='epoch')
	parser.add_argument('--batch_size', type=int, 			default=64, help='batch size')
	parser.add_argument('--start_lr', type=float, 			default=1e-05, help='start learning rate')
	parser.add_argument('--lr_update_interval', type=int, 	default=1, help='learning rate update interval')
	parser.add_argument('--lr_update_rate', type=float, 	default=0.9, help='learning rate update rate')

	parser.add_argument('--debug', type=bool, 				default=False, help='True or False')
	
	args = parser.parse_args()

	SRC_PATH 			= args.src_path
	SRC_FRAC 			= args.src_frac

	METHOD 				= args.method
	BITRATE 			= args.br + "k"
	LOG_PATH 			= args.log_path
	SUMMARY_INTERVAL 	= args.summary_interval
	CHECKPOINT_INTERVAL = args.checkpoint_interval

	NETWORK1 			= args.network1
	CHECKPOINT1_PATH 	= args.checkpoint1_path

	NETWORK2 			= args.network2
	CHECKPOINT2_PATH 	= args.checkpoint2_path

	NETWORK3 			= args.network3
	CHECKPOINT3_PATH 	= args.checkpoint3_path

	REG 				= args.regularizer
	
	EPOCHS 				= args.epoch
	BATCH_SIZE 			= args.batch_size
	START_LR 			= args.start_lr
	LR_UPDATE_INTERVAL 	= args.lr_update_interval
	LR_UPDATE_RATE 		= args.lr_update_rate

	DEBUG 				= args.debug
	
	print_args(args)




	################################################## Setup the training options
	# Load model
	NUM_CLASS = 4 if METHOD == "multi" else 2

	model1 = load_model(NETWORK1, REG, NUM_CLASS)
	model2 = load_model(NETWORK2, REG, NUM_CLASS)
	model3 = load_model(NETWORK3, REG, NUM_CLASS)

	load_ckpt(model1, CHECKPOINT1_PATH, START_LR)
	load_ckpt(model2, CHECKPOINT2_PATH, START_LR)
	load_ckpt(model3, CHECKPOINT3_PATH, START_LR)



	################################################## Setup the dataset
	# Set train, validation, and test data
	train_fnames = txt2list(glob(join(SRC_PATH, METHOD, "train_*.txt")))
	test_fnames = txt2list(glob(join(SRC_PATH, METHOD, "test_*.txt")))
	valid_fnames = txt2list(glob(join(SRC_PATH, METHOD, "valid_*.txt")))

	# Reduce the dataset
	train_fnames = train_fnames[:int(len(train_fnames) * SRC_FRAC)]
	test_fnames = test_fnames[:int(len(test_fnames) * SRC_FRAC)]
	valid_fnames = valid_fnames[:int(len(valid_fnames) * SRC_FRAC)]
	
	# Load data
	valid_dataset  	= configure_dataset(valid_fnames, BATCH_SIZE, shuffle=False)
	train_dataset 	= configure_dataset(train_fnames, BATCH_SIZE, shuffle=False)
	test_dataset 	= configure_dataset(test_fnames, BATCH_SIZE, shuffle=False)
	
	# Set callback functions 
	LOG_PATH, callbacks = load_callbacks(args)


	# write the argument information
	write_args(args, join(LOG_PATH, "setup.txt"))


	# Set number of steps per epoch
	total_train = 2 * len(train_fnames)
	total_valid = 2 * len(valid_fnames)
	total_test = 2 * len(test_fnames)

	STEPS_PER_EPOCH_TRAIN = total_train // BATCH_SIZE
	STEPS_PER_EPOCH_VALID = total_valid // BATCH_SIZE
	STEPS_PER_EPOCH_TEST = total_test // BATCH_SIZE

	'''
	################################################## Save the model 1, 2 result
	# Extract inputs
	input_data = np.zeros((2 * total_test, 1024))
	input_label = np.zeros((2 * total_test, 4))

	offset = 0
	for frames, label in test_dataset:
		print_progress("process on {:4.4f}%", 100 * offset / total_test)
		unit = frames.shape[0]

		_, gpa1 = model1(frames)
		_, gpa2 = model2(frames)
		input_data[offset:offset+unit, :] = tf.concat([gpa1, gpa2], axis=-1)
		input_label[offset:offset+unit, :] = label

		offset += unit


	# Save numpy file
	np.save('./fusion/_test_data', input_data)
	np.save('./fusion/_test_label', input_label)
	'''
	
	
	################################################## Train the model 3
	train_dataset = configure_dataset_by_np('./fusion/train_data.npy', './fusion/train_label.npy', BATCH_SIZE)
	valid_dataset = configure_dataset_by_np('./fusion/valid_data.npy', './fusion/valid_label.npy', BATCH_SIZE)
	test_dataset = configure_dataset_by_np('./fusion/test_data.npy', './fusion/test_label.npy', BATCH_SIZE)

	history = model3.fit(train_dataset, \
						epochs=EPOCHS, \
						steps_per_epoch=           , \
						callbacks=callbacks, \
						validation_data=valid_dataset, \
						validation_steps=STEPS_PER_EPOCH_VALID, \
						verbose=1)

	write_history(history, join(LOG_PATH, "train.txt"))



	################################################## Test the model
	result = model3.evaluate(test_dataset, steps=STEPS_PER_EPOCH_TEST)
	
	write_result(["Loss", "Accuracy"], result, join(LOG_PATH, "test.txt"))


	







if __name__=="__main__":
	main()


