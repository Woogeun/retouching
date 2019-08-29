
import argparse
from glob import glob
from os import cpu_count
from os.path import join

import skvideo.io as vio
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from network.Networks_structure_srnet_keras import SRNet
from network.Networks_structure_mislnet_keras import MISLNet


def detect_frame(model, frame):
	result_accuracy = model.predict(np.array([frame]), verbose=0)

	return result_accuracy


def is_original(data):
	score = 0
	for datum in data:
		if datum[0, 1] >  0.8:
			score += 1

	return score < len(data) / 2 


def detect_video(model, video_name):
	# video read
	video_meta = vio.ffprobe(video_name)
	video = np.array(vio.vread(video_name, outputdict={"-pix_fmt": "gray"}))
	fn, w, h, c = video.shape
	if w != 256 or h != 256 or c != 1: 
		raise(BaseException("================ wrong size file: \"{}\"".format(fname)))


	# predict the video retouch tampering
	predicted_data = []
	for frame in video:
		predicted_data.append(detect_frame(model, frame))
	
	idx = 0
	for r in predicted_data:
		print("{}: {}".format(idx,r))
		idx += 1

	prediction = is_original(predicted_data)

	return prediction, predicted_data


def show_result(data):
	fn = range(len(data))
	plt.plot(fn, data)
	plt.show()


def main():
	################################################## Parse the arguments
	parser = argparse.ArgumentParser(description='Train retouch detection network.')
	parser.add_argument('--file_path', type=str, default='../train_strong/median/mclv_S_0_1_500_0_.mp4', help='test file path')
	parser.add_argument('--method', type=str, default="blur", help='method')
	parser.add_argument('--network', type=str, default="SRNet", help='SRNet or MISLNet')
	parser.add_argument('--network_scale', type=float, default=1.0, help='network scale')
	parser.add_argument('--regularizer', type=float, default=0.001, help='regularizer')
	parser.add_argument('--batch_size', type=int, default=4, help='batch size')
	parser.add_argument('--stack_size', type=int, default=1, help='stack size')
	parser.add_argument('--checkpoint_path', type=str, default="./logs/20190829_151426_noise/checkpoint/weights_0", help='checkpoint path')
	parser.add_argument('--checkpoint_method', type=str, default="not", help='latest or not')
	args = parser.parse_args()

	FILE_PATH 			= args.file_path
	METHOD 				= args.method
	NETWORK 			= args.network
	SCALE 				= args.network_scale
	REG 				= args.regularizer
	BATCH_SIZE 			= args.batch_size
	STACK_SIZE 			= args.stack_size
	CHECKPOINT_PATH 	= args.checkpoint_path
	CHECKPOINT_METHOD 	= args.checkpoint_method


	################################################## Load checkpoint file
	if NETWORK == 'SRNet':
		model = SRNet(SCALE, REG)
	elif NETWORK == 'MISLNet':
		model = MISLNet(SCALE, REG)


	checkpoint = CHECKPOINT_PATH
	if CHECKPOINT_METHOD == "latest":
		checkpoint = tf.train.latest_checkpoint(CHECKPOINT_PATH)


	model.build(input_shape=(None,256,256,1))
	model.summary()

	model.load_weights(checkpoint)
	print(model.layers[0].get_weights()[0])
	



	################################################## Predict the data
	prediction, predicted_data = detect_video(model, FILE_PATH)
	# show_result(predicted_data)
	print("Original: ", prediction)
	







if __name__ == "__main__":
	main()


