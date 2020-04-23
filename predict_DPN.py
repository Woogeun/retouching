"""
DPN prediction module
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




def main():
	################################################## Parse the arguments
	parser = argparse.ArgumentParser(description='Train retouch detection network.')
	parser.add_argument('--src_path', type=str, 			default='./samples/blur', help='test source path')
	parser.add_argument('--dst_path', type=str, 			default='../train_strong/median/', help='image save destination path')

	parser.add_argument('--network1', type=str, 			default="SRNet", help='SRNet or MISLNet or DCTNet or MesoNet')
	parser.add_argument('--checkpoint1_path', type=str, 	default="./logs/20191014_142248_multi_SRNet_93/checkpoint/weights_10", help='checkpoint1 path')

	parser.add_argument('--network2', type=str, 			default="DCTNet", help='SRNet or MISLNet or DCTNet or MesoNet')
	parser.add_argument('--checkpoint2_path', type=str, 	default="./logs/20191016_083522_multi_DCTNet_88/checkpoint/weights_14", help='checkpoint2 path')

	parser.add_argument('--network3', type=str, 			default="Fusion", help='SRNet or MISLNet or DCTNet or MesoNet')
	parser.add_argument('--checkpoint3_path', type=str, 	default="", help='checkpoint3 path')

	parser.add_argument('--regularization', type=float, 	default=0.0001, help='regularization term')
	parser.add_argument('--method', type=str, 				default="multi", help='blur, median, noise or multi')

	args = parser.parse_args()

	SRC_PATH 	= args.src_path
	DST_PATH 	= args.dst_path

	NETWORK1 			= args.network1
	CHECKPOINT1_PATH 	= args.checkpoint1_path

	NETWORK2 			= args.network2
	CHECKPOINT2_PATH 	= args.checkpoint2_path

	NETWORK3 			= args.network3
	CHECKPOINT3_PATH 	= args.checkpoint3_path

	REG 		= args.regularization
	METHOD 		= args.method

	print_args(args)



	################################################## Load model with checkpoint
	NUM_CLASS = 4 if METHOD == "multi" else 2

	model1 = load_model(NETWORK1, REG, NUM_CLASS)
	model2 = load_model(NETWORK2, REG, NUM_CLASS)
	model3 = load_model(NETWORK3, REG, NUM_CLASS)

	load_ckpt(model1, CHECKPOINT1_PATH)
	load_ckpt(model2, CHECKPOINT3_PATH)
	load_ckpt(model3, CHECKPOINT3_PATH)



	################################################## Predict the data
	detector = Detector(model1=model1, model2=model2, model3=model3, dst_path=DST_PATH)

	if isdir(SRC_PATH):
		fnames = glob(join(SRC_PATH, "*.mp4"))
		for idx, fname in enumerate(fnames):
			print("*********************{}: {}".format(idx, fname))
			predicted_data = detector.detect_video(fname, is_show=False)

	else:
		predicted_data = detector.detect_video(SRC_PATH, is_show=True)

	







if __name__ == "__main__":
	main()


