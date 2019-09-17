"""
Video retouch detection training module
@authorized by Shasha Bae
@description: train the model for retouch detection
"""

import argparse
from os.path import join
from glob import glob

import tensorflow as tf
from tensorflow.python import keras
from keras import backend as K
from sklearn.model_selection import train_test_split

from network.Networks_structure_srnet_keras import SRNet 
from network.Networks_structure_mislnet_keras import MISLNet
from network.Networks_structure_namnet_keras import NamNet
from network.Networks_functions_keras import configure_dataset, load_callbacks, print_args, write_args, write_history, write_result




def main():

	################################################## parse the arguments
	parser = argparse.ArgumentParser(description='Train retouch detection network.')
	# parser.add_argument('--train_path', type=str, default='./retouch_tfrecord_train', help='train dataset path')
	# parser.add_argument('--test_path', type=str, default='./retouch_tfrecord_test', help='test dataset path')
	parser.add_argument('--src_path', type=str, default='./tfrecord_retouch_strong', help='source path')
	parser.add_argument('--fraction', type=float, default=0.2, help='fraction of validation dataset')
	parser.add_argument('--method', type=str, default="multi", help='blur, median, noise or multi')
	parser.add_argument('--br', type=str, default="*", help='bitrate')
	parser.add_argument('--log_path', type=str, default='./logs', help='log path')
	parser.add_argument('--batch_size', type=int, default=4, help='batch size')
	parser.add_argument('--network', type=str, default="INPUT_NETWORK", help='SRNet or MISLNet or NamNet')
	parser.add_argument('--network_scale', type=float, default=1.0, help='network scale')
	parser.add_argument('--regularizer', type=float, default=0.001, help='regularizer')
	parser.add_argument('--epoch', type=int, default=3, help='epoch')
	parser.add_argument('--summary_interval', type=int, default=1, help='loss inverval')
	parser.add_argument('--checkpoint_interval', type=int, default=1, help='checkpoint interval')
	parser.add_argument('--start_lr', type=float, default=1e-03, help='start learning rate')
	parser.add_argument('--lr_update_interval', type=int, default=1, help='learning rate update interval')
	parser.add_argument('--lr_update_rate', type=float, default=0.9, help='learning rate update rate')
	
	args = parser.parse_args()

	# TRAIN_PATH 			= args.train_path
	# TEST_PATH 			= args.test_path
	SRC_PATH 			= args.src_path
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
	

	print_args(args)



	################################################## Setup the dataset
	# Set train, validation, and test data
	total_fnames = glob(join(SRC_PATH, METHOD, BITRATE, "*.tfrecord"))
	train_fnames, test_fnames = train_test_split(total_fnames, test_size=0.2, random_state=123)
	train_fnames, valid_fnames = train_test_split(train_fnames, test_size=FRACTION, random_state=321)


	# Load data
	valid_dataset  	= configure_dataset(valid_fnames, BATCH_SIZE)
	train_dataset 	= configure_dataset(train_fnames, BATCH_SIZE)
	test_dataset 	= configure_dataset(test_fnames, BATCH_SIZE)
	


	################################################## Setup the training options
	# Load model
	if NETWORK == "SRNet":
		model = SRNet(SCALE, REG)
	elif NETWORK == "MISLNet":
		model = MISLNet(SCALE, REG)
	elif NETWORK == "NamNet":
		model = NamNet(SCALE, REG)
	else:
		raise(BaseException("No such network: {}".format(NETWORK)))


	# Setup train options
	optimizer = tf.keras.optimizers.Adamax(lr=START_LR)
	loss = 'categorical_crossentropy'
	metrics = {	"Accuracy": tf.keras.metrics.CategoricalAccuracy()}
	# metrics = {	"Accuracy": tf.keras.metrics.CategoricalAccuracy(), \
				# "True Positive": tf.keras.metrics.TruePositives(), \
				# "True Negative": tf.keras.metrics.TrueNegatives(), \
				# "False Positive": tf.keras.metrics.FalsePositives(), \
				# "False Negative": tf.keras.metrics.FalseNegatives()}
				# "SensitivityAtSpecificity": tf.keras.metrics.SensitivityAtSpecificity(specificity=0.5, num_thresholds=1)}

	model.compile(optimizer=optimizer, loss=loss, metrics=list(metrics.values()))
	model.build(input_shape=(None,256,256,1))
	model.summary()
	model.load_weights("./logs/20190902_154638_noise_93/checkpoint/weights_29")



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


