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
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras

from network import Networks_functions_keras
from network.Networks_functions_keras import configure_dataset
from network.Networks_structure_srnet_keras import SRNet
from network.Networks_structure_mislnet_keras import MISLNet
from network.Networks_structure_dctnet_keras import DCTNet

tf.enable_eager_execution()
Networks_functions_keras.NUM_CLASS= 4
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
	

def dataset2numpy(dataset, num_files, batch_size, steps):
	labels = np.zeros((num_files, 4))
	iterator = dataset.make_one_shot_iterator()
	next_val = iterator.get_next()
	with tf.Session() as sess:
		for offset in range(0, num_files, batch_size):
			_, label = sess.run(next_val)
			# print(label)
			# labels[offset:offset+batch_size] = np.argmax(label, axis=-1)
			labels[offset:offset+batch_size, :] = label
			print("{}/{}".format(offset, num_files))

	return labels

def txt2list(txts):
	fnames = []
	for txt in txts:
		with open(txt, 'r') as f:
			fnames += f.read().splitlines()

	return fnames


def main():
	################################################## Parse the arguments
	parser = argparse.ArgumentParser(description='Train retouch detection network.')
	parser.add_argument('--src_path', type=str, default='E:\\paired_minibatch', help='test source path')
	parser.add_argument('--train_path', type=str, 			default='./train_*.txt', help='source path')
	parser.add_argument('--test_path', type=str, 			default='./test_*.txt', help='source path')
	parser.add_argument('--validation_path', type=str, 		default='./validation_*.txt', help='source path')

	parser.add_argument('--net1', type=str, default="SRNet", help='SRNet or MISLNet or NamNet or MMCNet or DCTNet or MesoNet')
	parser.add_argument('--net1_scale', type=float, default=1.0, help='network scale')
	parser.add_argument('--net1_cktp', type=str, default="./logs/20191014_142248_multi_SRNet_93/checkpoint/weights_10", help='checkpoint path')
	parser.add_argument('--net2', type=str, default="DCTNet", help='SRNet or MISLNet or NamNet or MMCNet or DCTNet or MesoNet')
	parser.add_argument('--net2_scale', type=float, default=1.0, help='network scale')
	parser.add_argument('--net2_cktp', type=str, default="./logs/20191016_083522_multi_DCTNet_88/checkpoint/weights_14", help='checkpoint path')
	
	parser.add_argument('--batch_size', type=int, default=128, help='batch size')
	parser.add_argument('--method', type=str, default="multi", help='blur median noise multi')
	parser.add_argument('--weight', type=float, default=0.5, help='weight of network 1')

	
	args = parser.parse_args()

	SRC_PATH 			= args.src_path
	TRAIN_PATH 			= args.train_path
	TEST_PATH 			= args.test_path
	VALIDATION_PATH 	= args.validation_path

	NET1 				= args.net1
	NET1_SCALE			= args.net1_scale
	NET1_CKTP 			= args.net1_cktp
	NET2 				= args.net2
	NET2_SCALE			= args.net2_scale
	NET2_CKTP 			= args.net2_cktp
	BATCH_SIZE 			= args.batch_size
	METHOD 				= args.method
	WEIGHT 				= args.weight



	################################################## Load the test files
	# Set test data
	train_fnames = txt2list(glob(TRAIN_PATH))
	test_fnames = txt2list(glob(TEST_PATH))
	valid_fnames = txt2list(glob(VALIDATION_PATH))
	train_fnames = train_fnames[:int(len(train_fnames) * 8 / 9)]


	fnames = test_fnames
	session = 'test'
	dir_path = './fusion'


	# Load data
	#dataset 	= configure_dataset(fnames, BATCH_SIZE, shuffle=False)
	

	dataset = tf.data.TFRecordDataset(fnames)
	dataset = dataset.map(Networks_functions_keras._parse_function, num_parallel_calls=cpu_count())

	################################################## Setup the training options
	# Load model
	model1 = load_model(NET1, NET1_SCALE, METHOD)
	model2 = load_model(NET2, NET2_SCALE, METHOD)


	# load the model weights
	load_cktp(model1, NET1_CKTP)
	load_cktp(model2, NET2_CKTP)
	


	################################################## Test the model
	STEPS_TEST = len(fnames) * 2 // BATCH_SIZE
	#result result_ = model1.evaluate(dataset, steps=STEPS_TEST)
	# read true label
	#labels = dataset2numpy(dataset, num_files=len(fnames) * 2, batch_size=BATCH_SIZE, steps=STEPS_TEST)
	#np.save(join(dir_path, 'label_{}.npy'.format(session)), labels)
	

	# calculate predicted data
	for frames, label in dataset:
		result = model1(frames)
		result2 = model1(frames)


	result1 = model1.predict(dataset, steps=STEPS_TEST, verbose=1)
	#np.save(join(dir_path, 'result_{}_{}.npy'.format(session, NET1)), result1)
	result2 = model2.predict(dataset, steps=STEPS_TEST, verbose=1)
	#np.save(join(dir_path, 'result_{}_{}.npy'.format(session, NET2)), result2)





	"""
	
	result1 = np.load('./fusion/result_train_SRNet.npy')
	result2 = np.load('./fusion/result_train_DCTNet.npy')
	labels_ = np.load('./fusion/label_train.npy')
	labels  = np.argmax(labels_, axis=-1)

	assert(result1.shape == result2.shape)
	assert(result1.shape[0] == labels.shape[0])

	for weight in range(0, 11, 1):
		weight *= 0.1
		result = result1 * weight + result2 * (1 - weight)
		result = np.argmax(result, axis=-1)
		accuracy = np.mean(np.abs(result == labels))
		print("weight: %.1f, accuracy: %.4f" % (weight, accuracy))
	"""

	










if __name__ == "__main__":
	main()


