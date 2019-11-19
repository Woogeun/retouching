"""
DPN evaluate module
@authorized by Shasha Bae
@description: evaluate the fused model which combined given two pre-trained models 
"""

import argparse
from os.path import join

import numpy as np

import tensorflow as tf

from util import *
from network import *




def main():
	################################################## Parse the arguments
	parser = argparse.ArgumentParser(description='Train retouch detection network.')
	parser.add_argument('--src_path', type=str, 	default='./split_names', help='source path')

	parser.add_argument('--net1', type=str, 		default="SRNet", help='SRNet or MISLNet or DCTNet or MesoNet')
	parser.add_argument('--net1_scale', type=float, default=1.0, help='network scale')
	parser.add_argument('--net1_cktp', type=str, 	default="./logs/20191014_142248_multi_SRNet_93/checkpoint/weights_10", help='checkpoint path')

	parser.add_argument('--net2', type=str, 		default="DCTNet", help='SRNet or MISLNet or NamNet or MMCNet or DCTNet or MesoNet')
	parser.add_argument('--net2_scale', type=float, default=1.0, help='network scale')
	parser.add_argument('--net2_cktp', type=str, 	default="./logs/20191016_083522_multi_DCTNet_88/checkpoint/weights_14", help='checkpoint path')

	parser.add_argument('--net3', type=str, 		default="Fusion", help='SRNet or MISLNet or NamNet or MMCNet or DCTNet or MesoNet')
	parser.add_argument('--net3_scale', type=float, default=1.0, help='network scale')
	parser.add_argument('--net3_cktp', type=str, 	default="-", help='checkpoint path')
	
	parser.add_argument('--batch_size', type=int, 	default=128, help='batch size')
	parser.add_argument('--method', type=str, 		default="multi", help='blur median noise multi')

	args = parser.parse_args()

	SRC_PATH 			= args.src_path

	NET1 				= args.net1
	NET1_SCALE			= args.net1_scale
	NET1_CKTP 			= args.net1_cktp

	NET2 				= args.net2
	NET2_SCALE			= args.net2_scale
	NET2_CKTP 			= args.net2_cktp

	NET3 				= args.net3
	NET3_SCALE			= args.net3_scale
	NET3_CKTP 			= args.net3_cktp

	BATCH_SIZE 			= args.batch_size
	METHOD 				= args.method

	print_args(args)



	################################################## Load the test files
	# Set test data
	test_fnames = txt2list(glob(join(SRC_PATH, METHOD, "test_*.txt")))
	dataset 	= configure_dataset(test_fnames, BATCH_SIZE)

	

	################################################## Setup the training options
	# Load model
	model1 = load_model(NET1, NET1_SCALE, METHOD)
	model2 = load_model(NET2, NET2_SCALE, METHOD)
	model3 = load_model(NET3, NET3_SCALE, METHOD)


	# load the model weights
	load_cktp(model1, NET1_CKTP)
	load_cktp(model2, NET2_CKTP)
	load_cktp(model3, NET3_CKTP)
	


	################################################## Test the model
	STEPS_TEST = 2 * len(test_fnames) // BATCH_SIZE

	input_data = []
	input_label = []


	# calculate predicted data
	for frames, label in dataset:
		_, gpa1 = model1(frames)
		_, gpa2 = model2(frames)
		input_data += tf.concat([gpa1, gpa2], axis=-1)
		input_label += label


	result = model3.evaluate(input_data, input_label, steps=STEPS_TEST, verbose=1)






if __name__ == "__main__":
	main()


