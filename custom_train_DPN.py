"""
SPN train module
@authorized by Shasha Bae
@description: train the SPN model
"""

import random
import argparse
import os
from os.path import join
from glob import glob

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import backend as K

from utils import *
from network import *




def train_one_batch_(inputs, labels, 
					model1, model2, model3, 
					loss_fn, optimizer, loss, accuracy):
	_, gap1 = model1(inputs)
	_, gap2 = model2(inputs)
	concat = tf.concat([gap1, gap2], axis=-1)
	with tf.GradientTape() as tape:
		# Calculate loss
		predictions = model3(concat)
		pred_loss = loss_fn(labels, predictions)
		total_loss = pred_loss

	# Update gradient
	trainable_variables = model3.trainable_variables
	gradients = tape.gradient(total_loss, trainable_variables)
	optimizer.apply_gradients(zip(gradients, trainable_variables))

	# Update loss and accuracy
	loss.update_state(total_loss)
	accuracy.update_state(labels, predictions)


def train_one_batch(inputs, labels, 
					model1, model2, model3, 
					loss_fn, optimizer, loss, accuracy):
	with tf.GradientTape() as tape:
		# Calculate loss
		pred1, gap1 = model1(inputs)
		pred2, gap2 = model2(inputs)
		concat = tf.concat([gap1, gap2], axis=-1)
		pred3 = model3(concat)

		reg_loss1 = tf.math.add_n(model1.losses)
		reg_loss2 = tf.math.add_n(model2.losses)
		# pred_loss1 = loss_fn(labels, pred1)
		# pred_loss2 = loss_fn(labels, pred2)
		pred_loss3 = loss_fn(labels, pred3)
		# total_loss = pred_loss1 + pred_loss2 + pred_loss3 + reg_loss1 + reg_loss2
		total_loss = pred_loss3 + reg_loss1 + reg_loss2

	# Update gradient
	trainable_variables = model1.trainable_variables\
						+ model2.trainable_variables\
						+ model3.trainable_variables
	gradients = tape.gradient(total_loss, trainable_variables)
	optimizer.apply_gradients(zip(gradients, trainable_variables))

	# Update loss and accuracy
	loss.update_state(total_loss)
	accuracy.update_state(labels, pred3)


def test_one_batch_(inputs, labels, 
					model1, model2, model3,
					loss_fn, loss, accuracy):
	# Calculate loss
	_, gap1 = model1(inputs)
	_, gap2 = model2(inputs)
	concat = tf.concat([gap1, gap2], axis=-1)
	predictions = model3(concat)

	pred_loss = loss_fn(labels, predictions)
	total_loss = pred_loss

	# Update loss and accuracy
	loss.update_state(total_loss)
	accuracy.update_state(labels, predictions)


def test_one_batch(inputs, labels, 
					model1, model2, model3,
					loss_fn, loss, accuracy):
	# Calculate loss
	_, gap1 = model1(inputs)
	_, gap2 = model2(inputs)
	concat = tf.concat([gap1, gap2], axis=-1)
	predictions = model3(concat)

	reg_loss1 = tf.math.add_n(model1.losses)
	reg_loss2 = tf.math.add_n(model2.losses)
	pred_loss = loss_fn(labels, predictions)
	total_loss = pred_loss + reg_loss1 + reg_loss2

	# Update loss and accuracy
	loss.update_state(total_loss)
	accuracy.update_state(labels, predictions)



def main():
	################################################## parse the arguments
	parser = argparse.ArgumentParser(description='Train retouch detection network.')
	parser.add_argument('--src_path', type=str, 			default='./split_names', help='source path')
	parser.add_argument('--src_frac', type=float, 			default=1.0, help='amount of training dataset')
	# parser.add_argument('--src_frac', type=float, 			default=0.0005, help='amount of training dataset')

	parser.add_argument('--method', type=str, 				default="noise", help='blur, median, noise or multi')
	parser.add_argument('--log_path', type=str, 			default='logs', help='log path')

	parser.add_argument('--network1', type=str,             default="SRNet", help='SRNet or MISLNet or DCTNet or MesoNet')
	parser.add_argument('--checkpoint1_path', type=str,     default="./logs/20191111_114343_noise_SRNet_90/checkpoint/weights_10", help='checkpoint1 path')
	# parser.add_argument('--checkpoint1_path', type=str,     default="", help='checkpoint1 path')

	parser.add_argument('--network2', type=str,             default="DCTNet", help='SRNet or MISLNet or DCTNet or MesoNet')
	parser.add_argument('--checkpoint2_path', type=str,     default="./logs/20200305_133911_noise_DCTNet_85/checkpoint/weights_5", help='checkpoint2 path')
	# parser.add_argument('--checkpoint2_path', type=str,     default="", help='checkpoint2 path')

	parser.add_argument('--network3', type=str,             default="Fusion", help='SRNet or MISLNet or DCTNet or MesoNet')
	parser.add_argument('--checkpoint3_path', type=str,     default="./logs/20200324_163146_noise_Fusion_90/checkpoint/weights_5", help='checkpoint2 path')
	# parser.add_argument('--checkpoint3_path', type=str,     default="", help='checkpoint3 path')
	
	parser.add_argument('--regularizer', type=float, 		default=0.0001, help='regularizer')
	
	parser.add_argument('--epoch', type=int, 				default=30, help='epoch')
	parser.add_argument('--batch_size', type=int, 			default=16, help='batch size')
	parser.add_argument('--start_lr', type=float, 			default=5e-5, help='start learning rate')
	parser.add_argument('--lr_update_rate', type=float, 	default=0.95, help='learning rate update rate')

	parser.add_argument('--debug', type=bool, 				default=False, help='True or False')

	args = parser.parse_args()

	SRC_PATH = args.src_path
	SRC_FRAC = args.src_frac

	METHOD 				= args.method
	LOG_PATH 			= args.log_path

	NETWORK1 			= args.network1
	CHECKPOINT1_PATH 	= args.checkpoint1_path

	NETWORK2 			= args.network2
	CHECKPOINT2_PATH 	= args.checkpoint2_path

	NETWORK3 			= args.network3
	CHECKPOINT3_PATH 	= args.checkpoint3_path

	REG = args.regularizer

	EPOCHS = args.epoch
	BATCH_SIZE = args.batch_size
	START_LR = args.start_lr
	LR_UPDATE_RATE = args.lr_update_rate

	DEBUG = args.debug

	print_args(args)


	################################################## Create directories
	log_path, ckpt_path, history, train_summary_writer, valid_summary_writer = load_logFiles(LOG_PATH, METHOD, NETWORK3)
	write_args(args, join(log_path, 'args.txt'))


	################################################## Setup the training options
	# Load model
	NUM_CLASS = 4 if METHOD == "multi" else 2
	model1 = load_model(NETWORK1, REG, NUM_CLASS, is_DPN=True)
	model2 = load_model(NETWORK2, REG, NUM_CLASS, is_DPN=True)
	model3 = load_model(NETWORK3, REG, NUM_CLASS)


	################################################## Setup the dataset
	# Load data
	train_dataset, train_total = configure_dataset(  SRC_PATH, METHOD, 'train', BATCH_SIZE, \
										shuffle=True, repeat=False, frac=SRC_FRAC)
	test_dataset, test_total = configure_dataset(   SRC_PATH, METHOD, 'test', BATCH_SIZE, \
    									shuffle=True, repeat=False, frac=SRC_FRAC)
	valid_dataset, valid_total = configure_dataset(  SRC_PATH, METHOD, 'valid', BATCH_SIZE, \
										shuffle=True, repeat=False, frac=SRC_FRAC)


	################################################## Setup learning environments
	# optimizer and loss function
	lr_schdule = tf.keras.optimizers.schedules.ExponentialDecay(
		START_LR,
		decay_steps=int(train_total / BATCH_SIZE),
		decay_rate=LR_UPDATE_RATE,
		staircase=True
	)

	optimizer = tf.keras.optimizers.Adamax(learning_rate=lr_schdule)
	loss_fn = tf.keras.losses.CategoricalCrossentropy()

	# Load checkpoint
	if CHECKPOINT1_PATH != "":
		model1.load_weights(CHECKPOINT1_PATH)
	if CHECKPOINT2_PATH != "":
		model2.load_weights(CHECKPOINT2_PATH)
	if CHECKPOINT3_PATH != "":
		model3.load_weights(CHECKPOINT3_PATH)


	################################################## Train the model
	train_loss = tf.keras.metrics.Mean(name='train_loss')
	train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
	valid_loss = tf.keras.metrics.Mean(name='valid_loss')
	valid_accuracy = tf.keras.metrics.CategoricalAccuracy(name='valid_accuracy')
	test_loss = tf.keras.metrics.Mean(name='test_loss')
	test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

	log_history(history, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 100)

	step = 1
	for epoch in range(EPOCHS):

		# Epoch initialization
		offset = 0
		train_loss.reset_states()
		train_accuracy.reset_states()
		valid_loss.reset_states()
		valid_accuracy.reset_states()

		# Train one epoch
		for inputs, labels in train_dataset:
			train_one_batch(inputs, labels,
							model1=model1, model2=model2, model3=model3,
							loss_fn=loss_fn, optimizer=optimizer,
							loss=train_loss, accuracy=train_accuracy)

			# Save tensorboard
			with train_summary_writer.as_default():
				tf.summary.scalar('loss', train_loss.result(), step=step)
				tf.summary.scalar('accuracy', train_accuracy.result(), step=step)

			# Log batch information
			offset += inputs.shape[0]
			step += 1
			log_one_batch("Train", offset, train_total, train_loss, train_accuracy, history)

		# Save checkpoint
		model1.save_weights(join(ckpt_path, f"weights_{epoch}_SRNet"), save_format='h5')
		model2.save_weights(join(ckpt_path, f"weights_{epoch}_DCTNet"), save_format='h5')
		model3.save_weights(join(ckpt_path, f"weights_{epoch}_Fusion"), save_format='h5')

		# Validation
		offset = 0
		for inputs, labels in valid_dataset:
			test_one_batch(inputs, labels,
						   model1=model1, model2=model2, model3=model3,
						   loss_fn=loss_fn,
						   loss=valid_loss, accuracy=valid_accuracy)

			# Log batch information
			offset += inputs.shape[0]
			log_one_batch("Validation", offset, valid_total, valid_loss, valid_accuracy, history)

		with valid_summary_writer.as_default():
			tf.summary.scalar('loss', valid_loss.result(), step=step)
			tf.summary.scalar('accuracy', valid_accuracy.result(), step=step)

		# Log epoch information
		log_history(history, f"[Epoch: {epoch:2d}] train_loss: {train_loss.result():.4f}, train_accuracy: {train_accuracy.result():.4f}%", 100)
		log_history(history, f"            valid_loss: {valid_loss.result():.4f}, valid_accuracy: {valid_accuracy.result():.4f}%\n", 100)
		log_history(history, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 100)

	# Test the model
	offset = 0
	for inputs, labels in test_dataset:
		test_one_batch(inputs, labels,
					   model1=model1, model2=model2, model3=model3,
					   loss_fn=loss_fn,
					   loss=test_loss, accuracy=test_accuracy)

		# Log batch information
		offset += inputs.shape[0]
		log_one_batch("Test", offset, test_total, test_loss, test_accuracy, history)

	history.close()


if __name__ == "__main__":
	K.clear_session()
	main()


