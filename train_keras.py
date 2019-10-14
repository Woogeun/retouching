"""
Video retouch detection training module
@authorized by Shasha Bae
@description: train the model for retouch detection
"""

import argparse
from os.path import join
from glob import glob
import random

import tensorflow as tf
from tensorflow.python import keras
from keras import backend as K
from sklearn.model_selection import train_test_split

from network.Networks_structure_srnet_keras import SRNet 
from network.Networks_structure_mislnet_keras import MISLNet
from network.Networks_structure_namnet_keras import NamNet
from network.Networks_structure_mmcnet_keras import MMCNet
from network.Networks_structure_mesonet_keras import MesoNet
from network.Networks_structure_dctnet_keras import DCTNet
from network import Networks_functions_keras
from network.Networks_functions_keras import configure_dataset, load_callbacks, print_args, write_args, write_history, write_result, ConfusionMatrix



def txt2list(txts):
	fnames = []
	for txt in txts:
		with open(txt, 'r') as f:
			fnames += f.read().splitlines()

	return fnames


def main():

	################################################## parse the arguments
	parser = argparse.ArgumentParser(description='Train retouch detection network.')
	parser.add_argument('--src_path', type=str, 			default='E:\\paired_minibatch', help='source path')
	parser.add_argument('--train_path', type=str, 			default='./train_*.txt', help='source path')
	parser.add_argument('--test_path', type=str, 			default='./test_*.txt', help='source path')
	parser.add_argument('--validation_path', type=str, 		default='./validation_*.txt', help='source path')

	parser.add_argument('--fraction', type=float, 			default=0.2, help='fraction of validation dataset')
	parser.add_argument('--method', type=str, 				default="multi", help='blur, median, noise or multi')
	parser.add_argument('--br', type=str, 					default="*", help='bitrate')
	parser.add_argument('--log_path', type=str, 			default='./logs', help='log path')
	parser.add_argument('--batch_size', type=int, 			default=32, help='batch size')
	parser.add_argument('--network', type=str, 				default="SRNet", help='SRNet or MISLNet or NamNet or MMCNet or DCTNet or MesoNet')
	parser.add_argument('--network_scale', type=float, 		default=1.0, help='network scale')
	parser.add_argument('--regularizer', type=float, 		default=0.0001, help='regularizer')
	parser.add_argument('--epoch', type=int, 				default=20, help='epoch')
	parser.add_argument('--summary_interval', type=int, 	default=1, help='loss inverval')
	parser.add_argument('--checkpoint_interval', type=int, 	default=1, help='checkpoint interval')
	parser.add_argument('--start_lr', type=float, 			default=1e-04, help='start learning rate')
	parser.add_argument('--lr_update_interval', type=int, 	default=1, help='learning rate update interval')
	parser.add_argument('--lr_update_rate', type=float, 	default=0.9, help='learning rate update rate')
	parser.add_argument('--debug', type=bool, 				default=False, help='True or False')
	args = parser.parse_args()

	SRC_PATH 			= args.src_path
	TRAIN_PATH 			= args.train_path
	TEST_PATH 			= args.test_path
	VALIDATION_PATH 	= args.validation_path

	FRACTION 			= args.fraction
	METHOD 				= args.method
	BITRATE 			= args.br + "k"
	LOG_PATH 			= args.log_path
	BATCH_SIZE 			= args.batch_size
	NETWORK 			= args.network
	SCALE 				= args.network_scale
	REG 				= args.regularizer
	EPOCHS 				= args.epoch
	SUMMARY_INTERVAL 	= args.summary_interval
	CHECKPOINT_INTERVAL = args.checkpoint_interval
	START_LR 			= args.start_lr
	LR_UPDATE_INTERVAL 	= args.lr_update_interval
	LR_UPDATE_RATE 		= args.lr_update_rate
	DEBUG 				= args.debug
	

	print_args(args)



	################################################## Setup the dataset
	# Set train, validation, and test data
	# strong_fnames = glob(join(SRC_PATH, "tfrecord_retouch_strong", METHOD, BITRATE, "*.tfrecord"))
	# weak_fnames = glob(join(SRC_PATH, "tfrecord_retouch_weak", METHOD, BITRATE, "*.tfrecord"))
	# total_fnames = []

	# for strong_fname, weak_fname in zip(strong_fnames, weak_fnames):
	# 	total_fnames += [strong_fname, weak_fname]

	# train_fnames, test_fnames = train_test_split(total_fnames, test_size=0.1, shuffle=False)
	# test_fnames, valid_fnames = train_test_split(test_fnames, test_size=0.5, shuffle=False)

	# random.shuffle(train_fnames)
	# train_fnames = train_fnames[:int(len(train_fnames) * 8 / 9)]
	# if DEBUG: train_fnames = train_fnames[:int(len(train_fnames) / 512)]
	# else: train_fnames = train_fnames[:int(len(train_fnames) / 2)]


	train_fnames = txt2list(glob(TRAIN_PATH))
	test_fnames = txt2list(glob(TEST_PATH))
	valid_fnames = txt2list(glob(VALIDATION_PATH))
	train_fnames = train_fnames[:int(len(train_fnames) * 8 / 9)]

	# print(len(train_fnames))
	# print(len(test_fnames))
	# print(len(valid_fnames))


	# Load data
	valid_dataset  	= configure_dataset(valid_fnames, BATCH_SIZE, is_color=(NETWORK == "MesoNet"))
	train_dataset 	= configure_dataset(train_fnames, BATCH_SIZE, is_color=(NETWORK == "MesoNet"))
	test_dataset 	= configure_dataset(test_fnames, BATCH_SIZE, is_color=(NETWORK == "MesoNet"))
	


	################################################## Setup the training options
	# Load model
	NUM_CLASS = 4 if METHOD == "multi" else 2
	Networks_functions_keras.NUM_CLASS = NUM_CLASS

	if NETWORK == "SRNet":
		model = SRNet(SCALE, REG, NUM_CLASS)
	elif NETWORK == "MISLNet":
		model = MISLNet(SCALE, REG, NUM_CLASS)
	elif NETWORK == "NamNet":
		model = NamNet(SCALE, REG, NUM_CLASS)
	elif NETWORK == "MMCNet":
		model = MMCNet(SCALE, REG, NUM_CLASS)
	elif NETWORK == "MesoNet":
		model = MesoNet(SCALE, REG, NUM_CLASS)
	elif NETWORK == "DCTNet":
		model = DCTNet(SCALE, REG, NUM_CLASS)
	else:
		raise(BaseException("No such network: {}".format(NETWORK)))


	# Setup train options
	optimizer = tf.keras.optimizers.Adamax(lr=START_LR)
	loss = 'categorical_crossentropy'
	metrics = {	"Accuracy": tf.keras.metrics.CategoricalAccuracy() ,\
				# "ConfusionMatrix": ConfusionMatrix(name="confusion_matrix", num_class=NUM_CLASS), \
				}

	model.compile(optimizer=optimizer, loss=loss, metrics=list(metrics.values()))
	model.build(input_shape=(None,256,256,1))
	model.summary()

	# For continuous learning
	model.load_weights("./logs/20191014_010419_multi/checkpoint/weights_9")



	################################################## Train the model
	# Set callback functions 
	LOG_PATH, callbacks = load_callbacks(args)


	# write the argument information
	write_args(args, join(LOG_PATH, "setup.txt"))


	# Set number of steps per epoch
	STEPS_PER_EPOCH_TRAIN = len(train_fnames) // BATCH_SIZE
	STEPS_PER_EPOCH_VALID = len(valid_fnames) // BATCH_SIZE


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
	STEPS_TEST = len(test_fnames) // BATCH_SIZE
	result = model.evaluate(test_dataset, steps=STEPS_TEST)
	
	write_result(["Loss"] + list(metrics.keys()), result, join(LOG_PATH, "test.txt"))










if __name__=="__main__":
	# For easy reset of notebook state.
	K.clear_session()  
	main()


