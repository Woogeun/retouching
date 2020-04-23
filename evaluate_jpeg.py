"""
SPN evaluate module for jpeg
@authorized by Shasha Bae
@description: evaluate the SPN model for jpeg images
"""

import argparse
from os.path import join, isdir
from glob import glob

import cv2
import random
import numpy as np

import tensorflow as tf

from utils import *
from network import *
from predictor import Detector



__DUMMY__ = 0.0001
def main():
	################################################## Parse the arguments
	parser = argparse.ArgumentParser(description='Train retouch detection network.')
	parser.add_argument('--src_path', type=str, 	default='../jpeg', help='source path')

	parser.add_argument('--net', type=str, 			default="SRNet", help='SRNet or MISLNet or DCTNet or MesoNet')
	parser.add_argument('--net_cktp', type=str, 	default="./logs/20191104_171914_blur_SRNet_98/checkpoint/weights_18", help='checkpoint path')
	# parser.add_argument('--net_cktp', type=str, 	default="./logs/20191107_134614_median_SRNet_98/checkpoint/weights_18", help='checkpoint path')
	# parser.add_argument('--net_cktp', type=str, 	default="./logs/20191111_114343_noise_SRNet_90/checkpoint/weights_35", help='checkpoint path')
	# parser.add_argument('--net_cktp', type=str, 	default="./logs/20191014_142248_multi_SRNet_93/checkpoint/weights_18", help='checkpoint path')
	# parser.add_argument('--net_cktp', type=str, 	default="./logs/20200306_143112_blur_Total_98/checkpoint/weights_29", help='checkpoint path')
	# parser.add_argument('--net_cktp', type=str, 	default="./logs/20200309_143754_median_Total_98/checkpoint/weights_21", help='checkpoint path')
	# parser.add_argument('--net_cktp', type=str, 	default="./logs/20200322_030005_noise_Total_90/checkpoint/weights_11", help='checkpoint path')
	
	parser.add_argument('--batch_size', type=int, 	default=64, help='batch size')
	parser.add_argument('--method', type=str, 		default="blur", help='blur median noise multi')
	# parser.add_argument('--method', type=str, 		default="median", help='blur median noise multi')
	# parser.add_argument('--method', type=str, 		default="noise", help='blur median noise multi')
	# parser.add_argument('--method', type=str, 		default="multi", help='blur median noise multi')

	args = parser.parse_args()

	SRC_PATH 			= args.src_path

	NET 				= args.net
	NET_CKTP 			= args.net_cktp

	BATCH_SIZE 			= args.batch_size
	METHOD 				= args.method

	print_args(args)


	################################################## Setup the training options
	# Load model
	NUM_CLASS = 4 if METHOD == "multi" else 2
	model = load_model(NET, __DUMMY__, NUM_CLASS)


	# load the model weights
	model.load_weights(NET_CKTP)



	################################################## Predict the data
	detector = Detector(model1=model, dst_path="../jpeg")

	fnames_negative = glob(join(SRC_PATH, "single", "*.jpg"))
	fnames_negative += glob(join(SRC_PATH, "single", "*.jpg"))
	fnames_positive = glob(join(SRC_PATH, METHOD, "strong", "*.jpg"))
	fnames_positive += glob(join(SRC_PATH, METHOD, "weak", "*.jpg"))

	total = 0
	correct = 0
	incorrect = 0
	single = 0
	manipulated = 0
	
	for idx, fname in enumerate(fnames_negative):
		frame = cv2.imread(fname)
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		predicted_data = detector.detect_frame(frame)[0, 0]
		if predicted_data > 0.5: 
			correct += 1
			single += 1
		else:
			incorrect += 1
			manipulated += 1

		print(f"{idx/len(fnames_negative) * 100:.2f}%")
		total += 1
		

	for idx, fname in enumerate(fnames_positive):
		frame = cv2.imread(fname)
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		predicted_data = detector.detect_frame(frame)[0, 1]
		if predicted_data > 0.5: 
			correct += 1
			manipulated += 1
		else:
			incorrect += 1
			single += 1

		print(f"{idx/len(fnames_positive) * 100:.2f}%")
		total += 1


	print(f"total: {total}, correct: {correct}, incorrect: {incorrect}, single: {single}, manipulated: {manipulated}")
	print(f"accuracy: {correct/total*100:.2f}%")



if __name__ == "__main__":
	main()