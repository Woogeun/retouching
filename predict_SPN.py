"""
SPN prediction module
@authorized by Shasha Bae
@description: predict whether the input video is tampered or not. If input type is directory, predict the videos in the directory 
"""

import argparse
from os.path import join, isdir
from glob import glob

import tensorflow as tf
from tensorflow import keras

from utils import *
from predictor import Detector




__DUMMY__ = 0.0001
def main():
	################################################## Parse the arguments
	parser = argparse.ArgumentParser(description='Train retouch detection network.')
	parser.add_argument('--src_path', type=str, 		default='./samples/blur', help='test source path')
	parser.add_argument('--dst_path', type=str, 		default='../train_strong/median/', help='image save destination path')

	parser.add_argument('--network', type=str, 			default="SRNet", help='SRNet or MISLNet or NamNet')
	parser.add_argument('--checkpoint', type=str, 		default="./logs/20191014_142248_multi_SRNet_93/checkpoint/weights_10", help='checkpoint path')

	parser.add_argument('--method', type=str, 			default="multi", help='blur, median, noise or multi')

	args = parser.parse_args()

	SRC_PATH 	= args.src_path
	DST_PATH 	= args.dst_path

	NETWORK 	= args.network
	CHECKPOINT 	= args.checkpoint

	METHOD 		= args.method

	print_args(args)



	################################################## Load model with checkpoint
	NUM_CLASS = 4 if METHOD == "multi" else 2
	model = load_model(NETWORK, __DUMMY__, NUM_CLASS)
	load_ckpt(model, CHECKPOINT)



	################################################## Predict the data
	detector = Detector(model1=model, dst_path=DST_PATH)

	if isdir(SRC_PATH):
		fnames = glob(join(SRC_PATH, "*.mp4"))
		for idx, fname in enumerate(fnames):
			print("*********************{}: {}".format(idx, fname))
			predicted_data = detector.detect_video(fname, is_show=False)

	else:
		predicted_data = detector.detect_video(SRC_PATH, is_show=True)

	







if __name__ == "__main__":
	main()


