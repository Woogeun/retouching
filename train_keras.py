
import argparse
from os.path import join
from glob import glob
from os import cpu_count

import tensorflow as tf
from tensorflow.python import keras
from keras import backend as K

from network.Networks_structure_srnet_keras import SRNet, SRNet_
from network.Networks_functions_keras import _parse_function, load_callbacks


def configure_dataset(fnames, batch_size):
	buffer_size = 16
	dataset = tf.data.TFRecordDataset(fnames)
	dataset = dataset.map(_parse_function, num_parallel_calls=cpu_count())
	dataset = dataset.prefetch(buffer_size=buffer_size) # recommend buffer_size = # of elements / batches
	dataset = dataset.shuffle(buffer_size=buffer_size) # recommend buffer_size = # of elements / batches
	dataset = dataset.repeat()
	dataset = dataset.batch(batch_size)

	return dataset



def main():

	################################################## parse the arguments
	parser = argparse.ArgumentParser(description='Train retouch detection network.')
	parser.add_argument('--train_path', type=str, default='./retouch_tfrecord_train', help='train dataset path')
	parser.add_argument('--test_path', type=str, default='./retouch_tfrecord_test', help='test dataset path')
	parser.add_argument('--fraction', type=float, default=0.2, help='fraction of validation dataset')
	parser.add_argument('--method', type=str, default="*", help='attack method')
	parser.add_argument('--br', type=str, default="*", help='bitrate')
	parser.add_argument('--log_path', type=str, default='./logs', help='log path')
	parser.add_argument('--batch_size', type=int, default=4, help='batch size')
	parser.add_argument('--epoch', type=int, default=3, help='epoch')
	parser.add_argument('--summary_interval', type=int, default=1, help='loss inverval')
	parser.add_argument('--checkpoint_interval', type=int, default=1, help='checkpoint interval')
	parser.add_argument('--start_lr', type=float, default=1e-03, help='start learning rate')
	parser.add_argument('--lr_update_interval', type=int, default=1, help='learning rate update interval')
	parser.add_argument('--lr_update_rate', type=float, default=0.9, help='learning rate update rate')
	parser.add_argument('--beta1', type=float, default=0.9, help='beta 1 for Adamax')
	parser.add_argument('--beta2', type=float, default=0.999, help='beta 2 for Adamax')
	args = parser.parse_args()

	TRAIN_PATH 			= args.train_path
	TEST_PATH 			= args.test_path
	FRACTION 			= args.fraction
	METHOD 				= args.method
	BITRATE 			= args.br + "br"
	LOG_PATH 			= args.log_path
	BATCH_SIZE 			= args.batch_size
	EPOCHS 				= args.epoch
	SUMMARY_INTERVAL 	= args.summary_interval
	CHECKPOINT_INTERVAL = args.checkpoint_interval
	START_LR 			= args.start_lr
	LR_UPDATE_INTERVAL 	= args.lr_update_interval
	LR_UPDATE_RATE 		= args.lr_update_rate
	BETA_1 				= args.beta1
	BETA_2 				= args.beta2



	################################################## Setup the checkpoint and dataset files
	# Set checkpoint 
	callbacks = load_callbacks(args)
	

	# Set train files
	whole_fnames 	= glob(join(TRAIN_PATH, METHOD, BITRATE, "*.tfrecord"))
	idx = int(len(whole_fnames) * FRACTION)
	
	valid_fnames 	= whole_fnames[:idx]
	train_fnames 	= whole_fnames[idx:]
	test_fnames 	= glob(join(TEST_PATH, METHOD, BITRATE, "*.tfrecord"))

	# Load data
	valid_dataset  	= configure_dataset(valid_fnames, BATCH_SIZE)
	train_dataset 	= configure_dataset(train_fnames, BATCH_SIZE)
	test_dataset 	= configure_dataset(test_fnames, BATCH_SIZE)
	





	################################################## Setup the training options
	# Load model
	model = SRNet()


	# Setup train options
	optimizer = tf.keras.optimizers.Adamax(lr=START_LR, beta_1=BETA_1, beta_2=BETA_2)
	loss = tf.keras.losses.CategoricalCrossentropy()
	loss = 'categorical_crossentropy'
	metrics = [tf.keras.metrics.CategoricalAccuracy()]
	
	# metrics = [tf.keras.metrics.CategoricalAccuracy(), \
	# 			tf.keras.metrics.TruePositives(), \
	# 			tf.keras.metrics.TrueNegatives(), \
	# 			tf.keras.metrics.FalsePositives(), \
	# 			tf.keras.metrics.FalseNegatives()]
	

	model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
	tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)


	# Learn the model
	STEPS_PER_EPOCH_TRAIN = len(train_fnames) // BATCH_SIZE
	STEPS_PER_EPOCH_VALID = len(valid_fnames) // BATCH_SIZE


	history = model.fit(train_dataset, \
						epochs=EPOCHS, \
						steps_per_epoch=STEPS_PER_EPOCH_TRAIN, \
						callbacks=callbacks, \
						validation_data=valid_dataset, \
						validation_steps=STEPS_PER_EPOCH_VALID)

	# Add train history log
	# print(history.history)


	# Test the model
	STEPS_TEST = len(test_fnames) // BATCH_SIZE
	result = model.evaluate(test_dataset, steps=STEPS_TEST)
	
	# Add test log
	# print(result)










if __name__=="__main__":
	# For easy reset of notebook state.
	K.clear_session()  
	main()


