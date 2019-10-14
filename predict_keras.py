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

from network.Networks_structure_srnet_keras import SRNet
from network.Networks_structure_mislnet_keras import MISLNet




class Detector():
	"""Class for the detecting whether the video is tampered or not."""
	def __init__(self, model, dst_path):
		self.model = model
		self.dst_path = dst_path

		try:
			makedirs(dst_path)
		except:
			pass



	def detect_frame(self, frame):
		"""Detect a framewise tampering

	   	# arguments
	   		frame: numpy array of one frame

		# Returns
			The result accuracy of the frame can be tampered. If frame is not tampered, result is closer to [1 0], else [0 1]
		"""

		result_accuracy = self.model.predict(np.array([frame]), verbose=0)

		return result_accuracy[0,1]


	def is_original(self, data):
		"""Determine the videowise tampering

	   	# arguments
	   		data: the list of the probability of frame can be tampered

		# Returns
			The bool that input video is not tampered. If the video is not tampered, returns True, else False.
		"""
		
		return False


	def detect_video(self, video_name, is_show=False):
		"""Detect a videowise tampering

	   	# arguments
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
			result = self.detect_frame(frame)
			predicted_data.append(result)

			if is_show:
				print("{}: {}".format(idx, result))
			

		# Determine the prediction result(original or not) and save figure
		prediction = self.is_original(predicted_data)
		fn = range(len(predicted_data))
		plt.plot(fn, predicted_data)
		# plt.savefig(join(self.dst_path, video_name.split('\\')[-1].split('.')[0] + 'png'))

		if is_show:
			plt.show()
			print("Original: ", prediction)

		plt.clf()

		return predicted_data



def main():
	################################################## Parse the arguments
	parser = argparse.ArgumentParser(description='Train retouch detection network.')
	parser.add_argument('--src_path', type=str, default='./samples/blur', help='test source path')
	parser.add_argument('--dst_path', type=str, default='../train_strong/median/', help='image save destination path')
	# parser.add_argument('--num_test', type=int, default=100, help='number of test videos in test directory')
	parser.add_argument('--network', type=str, default="SRNet", help='SRNet or MISLNet or NamNet')
	parser.add_argument('--network_scale', type=float, default=0.5, help='network scale')
	parser.add_argument('--checkpoint', type=str, default="20190914_092006_blur_98", help='checkpoint path')
	args = parser.parse_args()

	SRC_PATH 			= args.src_path
	DST_PATH 			= args.dst_path
	# NUM_TEST 			= args.num_test
	NETWORK 			= args.network
	SCALE 				= args.network_scale
	CHECKPOINT 			= join('./logs', args.checkpoint, 'checkpoint', 'weights_29')



	################################################## Load checkpoint file
	# load model
	if NETWORK == 'SRNet':
		model = SRNet(SCALE)
	elif NETWORK == 'MISLNet':
		model = MISLNet(SCALE)
	elif NETWORK == "NamNet":
		model = NamNet(SCALE)
	else:
		raise(BaseException("No such network: {}".format(NETWORK)))


	# load the model weights
	model.build(input_shape=(None,256,256,1))
	model.summary()
	model.load_weights(CHECKPOINT)
	


	################################################## Predict the data
	detector = Detector(model, DST_PATH)

	if isdir(SRC_PATH):
		fnames = glob(join(SRC_PATH, "*.mp4"))
		# random.shuffle(fnames)
		# fnames = fnames[:NUM_TEST]

		for idx, fname in enumerate(fnames):
			print("*********************{}: {}".format(idx, fname))
			predicted_data = detector.detect_video(fname, is_show=False)
			print(sum(predicted_data) / len(predicted_data))

	else:
		predicted_data = detector.detect_video(SRC_PATH, is_show=True)

	







if __name__ == "__main__":
	main()


