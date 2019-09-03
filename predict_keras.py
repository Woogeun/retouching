"""
Video retouch detection prediction module
@authorized by Shasha Bae
@description: predict whether the input video is tampered or not. If input type is directory, predict the videos in the directory 
"""

import argparse
from glob import glob
from os import cpu_count
from os.path import join, isdir
import random

import skvideo.io as vio
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from network.Networks_structure_srnet_keras import SRNet
from network.Networks_structure_mislnet_keras import MISLNet




def detect_frame(model, frame):
	"""Detect a framewise tampering

   	# arguments
   		model: trained tf.keras.models
   		frame: numpy array of one frame

	# Returns
		The result accuracy of the frame can be tampered. If frame is not tampered, result is closer to [1 0], else [0 1]
	"""

	result_accuracy = model.predict(np.array([frame]), verbose=0)

	return result_accuracy


def is_original(data):
	"""Determine the videowise tampering

   	# arguments
   		data: the list of the probability of frame can be tampered

	# Returns
		The bool that input video is not tampered. If the video is not tampered, returns True, else False.
	"""
	
	return False


def show_result(data):
	"""Show the graph of predicted data visually

   	# arguments
   		data: the list of the probability of frame can be tampered
	"""

	fn = range(len(data))
	plt.plot(fn, data)
	plt.show()


def detect_video(model, video_name, is_show=False):
	"""Detect a videowise tampering

   	# arguments
   		model: trained tf.keras.models
   		video_name: the string of video file name
   		is_show: the bool of whether show the result graph

	# Returns
		The tuple of whether the video is tampered and predicted data
	"""

	# Read video
	video_meta = vio.ffprobe(video_name)
	video = np.array(vio.vread(video_name, outputdict={"-pix_fmt": "gray"}))
	fn, w, h, c = video.shape
	if w != 256 or h != 256 or c != 1: 
		raise(BaseException("================ wrong size file: \"{}\"".format(fname)))


	# Predict the video retouch tampering
	predicted_data = []
	for idx, frame in enumerate(video):
		result = detect_frame(model, frame)
		predicted_data.append(result[0,1])

		if is_show:
			print("{}: {}".format(idx, result))
		

	prediction = is_original(predicted_data)


	if is_show:
		show_result(predicted_data)
		print("Original: ", prediction)

	return prediction, predicted_data





def main():
	################################################## Parse the arguments
	parser = argparse.ArgumentParser(description='Train retouch detection network.')
	parser.add_argument('--file_path', type=str, default='../train_strong/median/', help='test file path')
	parser.add_argument('--num_test', type=int, default=100, help='number of test videos in test directory')
	parser.add_argument('--method', type=str, default="blur", help='method')
	parser.add_argument('--network', type=str, default="INPUT_NETWORK", help='SRNet or MISLNet or NamNet')
	parser.add_argument('--network_scale', type=float, default=1.0, help='network scale')
	parser.add_argument('--regularizer', type=float, default=0.001, help='regularizer')
	parser.add_argument('--batch_size', type=int, default=4, help='batch size')
	parser.add_argument('--stack_size', type=int, default=1, help='stack size')
	parser.add_argument('--checkpoint', type=str, default="./logs/20190901_020912_median_95/checkpoint/weights_29", help='checkpoint path')
	args = parser.parse_args()

	FILE_PATH 			= args.file_path
	NUM_TEST 			= args.num_test
	METHOD 				= args.method
	NETWORK 			= args.network
	SCALE 				= args.network_scale
	REG 				= args.regularizer
	BATCH_SIZE 			= args.batch_size
	STACK_SIZE 			= args.stack_size
	CHECKPOINT 			= args.checkpoint



	################################################## Load checkpoint file
	# load model
	if NETWORK == 'SRNet':
		model = SRNet(SCALE, REG)
	elif NETWORK == 'MISLNet':
		model = MISLNet(SCALE, REG)
	elif NETWORK == "NamNet":
		model = NamNet(SCALE, REG)
	else:
		raise(BaseException("No such network: {}".format(NETWORK)))


	# load the model weights
	model.build(input_shape=(None,256,256,1))
	model.summary()
	model.load_weights(CHECKPOINT)
	


	################################################## Predict the data
	if isdir(FILE_PATH):
		total_result = []
		fnames = glob(join(FILE_PATH, "*.mp4"))
		random.shuffle(fnames)
		fnames = fnames[:NUM_TEST]

		for idx, fname in enumerate(fnames):
			print("*********************{}: {}".format(idx, fname))
			prediction, predicted_data = detect_video(model, fname, is_show=False)
			total_result += predicted_data
		print(np.mean(total_result))

	else:
		prediction, predicted_data = detect_video(model, FILE_PATH, is_show=False)
		print(np.mean(predicted_data))

	







if __name__ == "__main__":
	main()


