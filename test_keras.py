"""
Video retouch detection prediction module
@authorized by Shasha Bae
@description: predict whether the input video is tampered or not. If input type is directory, predict the videos in the directory 
"""

import argparse
from glob import glob
from os import cpu_count, makedirs
from os.path import join, isdir
import random

import skvideo.io as vio
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from network import Networks_functions_keras
from network.Networks_functions_keras import configure_dataset
from network.Networks_structure_srnet_keras import SRNet
from network.Networks_structure_mislnet_keras import MISLNet
from network.Networks_structure_dctnet_keras import DCTNet

def txt2list(txts):
	fnames = []
	for txt in txts:
		with open(txt, 'r') as f:
			fnames += f.read().splitlines()

	return fnames


def load_model(model_name, SCALE, METHOD):
	NUM_CLASS = 4 if METHOD == "multi" else 2
	Networks_functions_keras.NUM_CLASS = NUM_CLASS
	REG = 0.0001

	if model_name == "SRNet":
		model = SRNet(SCALE, REG, NUM_CLASS)
	elif model_name == "MISLNet":
		model = MISLNet(SCALE, REG, NUM_CLASS)
	elif model_name == "NamNet":
		model = NamNet(SCALE, REG, NUM_CLASS)
	elif model_name == "MMCNet":
		model = MMCNet(SCALE, REG, NUM_CLASS)
	elif model_name == "DCTNet":
		model = DCTNet(SCALE, REG, NUM_CLASS)
	elif model_name == "MesoNet":
		model = MesoNet(SCALE, REG, NUM_CLASS)
	else:
		raise(BaseException("No such network: {}".format(NETWORK)))

	return model


def load_cktp(model, cktp_path):
	START_LR = 0.0001
	optimizer = tf.keras.optimizers.Adamax(lr=START_LR)
	loss = 'categorical_crossentropy'
	metrics = {	"Accuracy": tf.keras.metrics.CategoricalAccuracy() }

	model.compile(optimizer=optimizer, loss=loss, metrics=list(metrics.values()))
	model.build(input_shape=(None,256,256,1))
	model.load_weights(cktp_path)



def main():
	################################################## Parse the arguments
	parser = argparse.ArgumentParser(description='Train retouch detection network.')
	parser.add_argument('--src_path', type=str, default='E:\\paired_minibatch', help='test source path')
	parser.add_argument('--train_path', type=str, 			default='./train_*.txt', help='source path')
	parser.add_argument('--test_path', type=str, 			default='./test_*.txt', help='source path')
	parser.add_argument('--validation_path', type=str, 		default='./validation_*.txt', help='source path')

	parser.add_argument('--fraction', type=int, default=0.1, help='number of test videos in test directory')
	parser.add_argument('--network', type=str, default="SRNet", help='SRNet or MISLNet or NamNet or MMCNet or DCTNet or MesoNet')
	parser.add_argument('--network_scale', type=float, default=1.0, help='network scale')
	parser.add_argument('--method', type=str, default='multi', help='blur, median, or multi')
	parser.add_argument('--regularizer', type=float, default=0.0001, help='regularizer')
	parser.add_argument('--start_lr', type=float, default=1e-04, help='start learning rate')
	parser.add_argument('--batch_size', type=int, default=32, help='batch size')
	parser.add_argument('--checkpoint', type=str, default="./logs/20191014_010419_multi/checkpoint/weights_9", help='checkpoint path')
	args = parser.parse_args()

	SRC_PATH 			= args.src_path
	TRAIN_PATH 			= args.train_path
	TEST_PATH 			= args.test_path
	VALIDATION_PATH 	= args.validation_path

	FRACTION 			= args.fraction
	NETWORK 			= args.network
	SCALE 				= args.network_scale
	METHOD 				= args.method
	REG 				= args.regularizer
	START_LR 			= args.start_lr
	CHECKPOINT 			= args.checkpoint
	BATCH_SIZE 			= args.batch_size


	################################################## Load the test files
	train_fnames = txt2list(glob(TRAIN_PATH))
	test_fnames = txt2list(glob(TEST_PATH))
	valid_fnames = txt2list(glob(VALIDATION_PATH))
	train_fnames = train_fnames[:int(len(train_fnames) * 8 / 9)]


	# Load data
	test_dataset 	= configure_dataset(test_fnames, BATCH_SIZE, is_color=(NETWORK == "MesoNet"))



	################################################## Setup the training options
	################################################## Setup the training options
	# Load model
	model = load_model(NETWORK, SCALE, METHOD)


	# load the model weights
	load_cktp(model, CHECKPOINT)
	


	################################################## Test the model
	STEPS_TEST = len(test_fnames) * 2 // BATCH_SIZE
	result = model.evaluate(test_dataset, steps=STEPS_TEST, verbose=1)







if __name__ == "__main__":
	main()


