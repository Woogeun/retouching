"""
SPN evaluate module
@authorized by Shasha Bae
@description: evaluate the SPN model 
"""

import argparse
from os.path import join
from glob import glob
from tqdm import tqdm

import numpy as np

import tensorflow as tf

from utils import *
from network import *




__DUMMY__ = 0.0001
def main():
	################################################## Parse the arguments
	parser = argparse.ArgumentParser(description='Train retouch detection network.')
	parser.add_argument('--src_path', type=str, 	default='./split_names', help='source path')

	parser.add_argument('--net', type=str, 			default="MesoNet", help='SRNet or MISLNet or DCTNet or MesoNet')
	# parser.add_argument('--net_cktp', type=str, 	default="./logs/20191104_171914_blur_SRNet_98/checkpoint/weights_18", help='checkpoint path')
	# parser.add_argument('--net_cktp', type=str, 	default="./logs/20191107_134614_median_SRNet_98/checkpoint/weights_18", help='checkpoint path')
	# parser.add_argument('--net_cktp', type=str, 	default="./logs/20191111_114343_noise_SRNet_90/checkpoint/weights_18", help='checkpoint path')
	# parser.add_argument('--net_cktp', type=str, 	default="./logs/20191014_142248_multi_SRNet_93/checkpoint/weights_18", help='checkpoint path')
	# parser.add_argument('--net_cktp', type=str, 	default="./logs/20191016_083522_multi_DCTNet_88/checkpoint/weights_14", help='checkpoint path')
	parser.add_argument('--net_cktp', type=str, 	default="./logs/20191016_140819_multi_MesoNet_60/checkpoint/weights_18", help='checkpoint path')
	# parser.add_argument('--net_cktp', type=str, default="./logs/20200130_151833_multi_XceptionNet_80/checkpoint/weights_18", help='checkpoint path')
	
	parser.add_argument('--batch_size', type=int, 	default=64, help='batch size')
	# parser.add_argument('--method', type=str, 		default="blur", help='blur median noise multi')
	# parser.add_argument('--method', type=str, 		default="median", help='blur median noise multi')
	# parser.add_argument('--method', type=str, 		default="noise", help='blur median noise multi')
	parser.add_argument('--method', type=str, 		default="multi", help='blur median noise multi')

	parser.add_argument('--br', type=str, 			default="500k", help='500k, 600k, 700k or 800k')

	args = parser.parse_args()

	SRC_PATH 			= args.src_path

	NET 				= args.net
	NET_CKTP 			= args.net_cktp

	BATCH_SIZE 			= args.batch_size
	METHOD 				= args.method

	BITRATE 			= args.br

	print_args(args)



	################################################## Setup the training options
	# Load model
	NUM_CLASS = 4 if METHOD == "multi" else 2
	model = load_model(NET, __DUMMY__, NUM_CLASS)


	# load the model weights
	load_ckpt(model, NET_CKTP, __DUMMY__)
	


	################################################## Load the test files
	# Set test data
	test_fnames = txt2list(glob(join(SRC_PATH, METHOD, "test_*.txt")))
	test_fnames = list(filter(lambda x: BITRATE in x, test_fnames))
	dataset 	= configure_dataset(test_fnames, BATCH_SIZE, shuffle=False)

	

	################################################## Test the model
	total_length = 2 * len(test_fnames)
	STEPS_TEST = total_length // BATCH_SIZE
	
	result = model.evaluate(dataset, steps=STEPS_TEST, verbose=1)




if __name__ == "__main__":
	main()


