"""
SPN evaluate module
@authorized by Shasha Bae
@description: evaluate the SPN model 
"""

import argparse

import numpy as np

import tensorflow as tf

from util import *
from network import *




def main():
	################################################## Parse the arguments
	parser = argparse.ArgumentParser(description='Train retouch detection network.')
	parser.add_argument('--test_path', type=str, 	default='./test_*.txt', help='source path')

	parser.add_argument('--net', type=str, 			default="SRNet", help='SRNet or MISLNet or DCTNet or MesoNet')
	parser.add_argument('--net_scale', type=float, 	default=1.0, help='network scale')
	parser.add_argument('--net_cktp', type=str, 	default="./logs/20191014_142248_multi_SRNet_93/checkpoint/weights_10", help='checkpoint path')
	
	parser.add_argument('--batch_size', type=int, 	default=128, help='batch size')
	parser.add_argument('--method', type=str, 		default="multi", help='blur median noise multi')

	args = parser.parse_args()

	TEST_PATH 			= args.test_path

	NET 				= args.net
	NET_SCALE			= args.net_scale
	NET_CKTP 			= args.net_cktp

	BATCH_SIZE 			= args.batch_size
	METHOD 				= args.method

	print_args(args)



	################################################## Load the test files
	# Set test data
	test_fnames = txt2list(glob(TEST_PATH))
	dataset 	= configure_dataset(test_fnames, BATCH_SIZE)

	

	################################################## Setup the training options
	# Load model
	model = load_model(NET, NET_SCALE, METHOD)


	# load the model weights
	load_cktp(model, NET_CKTP)
	


	################################################## Test the model
	STEPS_TEST = 2 * len(test_fnames) // BATCH_SIZE

	result = model1.evaluate(dataset, steps=STEPS_TEST, verbose=1)






if __name__ == "__main__":
	main()


