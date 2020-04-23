"""
SPN evaluate module
@authorized by Shasha Bae
@description: evaluate the SPN model 
"""

import argparse
from os.path import join
from glob import glob
from tqdm import tqdm
from tensorflow.python.keras import Input, Sequential
from tensorflow.python.keras.models import Model
import numpy as np

import tensorflow as tf

from utils import *
from network import *




__DUMMY__ = 0.0001
def main():
	################################################## Parse the arguments
	parser = argparse.ArgumentParser(description='Train retouch detection network.')
	parser.add_argument('--src_path', type=str, 	default='./split_names', help='source path')

	parser.add_argument('--net1', type=str, default="SRNet", help='SRNet or MISLNet or DCTNet or MesoNet')
	parser.add_argument('--net1_cktp', type=str, default="./logs/20191014_142248_multi_SRNet_93/checkpoint/weights_10",
						help='checkpoint path')

	parser.add_argument('--net2', type=str, default="DCTNet", help='SRNet or MISLNet or NamNet or MMCNet or DCTNet or MesoNet')
	parser.add_argument('--net2_cktp', type=str, default="./logs/20191016_083522_multi_DCTNet_88/checkpoint/weights_14", help='checkpoint path')

	parser.add_argument('--net3', type=str, default="Fusion", help='SRNet or MISLNet or NamNet or MMCNet or DCTNet or MesoNet')
	parser.add_argument('--net3_cktp', type=str, default="./logs/20191203_140056_multi_Fusion_96/checkpoint/weights_10", help='checkpoint path')
	
	parser.add_argument('--batch_size', type=int, 	default=1, help='batch size')
	# parser.add_argument('--method', type=str, 		default="blur", help='blur median noise multi')
	# parser.add_argument('--method', type=str, 		default="median", help='blur median noise multi')
	# parser.add_argument('--method', type=str, 		default="noise", help='blur median noise multi')
	parser.add_argument('--method', type=str, 		default="multi", help='blur median noise multi')

	parser.add_argument('--br', type=str, 			default="500k", help='500k, 600k, 700k or 800k')	

	args = parser.parse_args()

	SRC_PATH 			= args.src_path

	NET1 = args.net1
	NET1_CKTP = args.net1_cktp

	NET2 = args.net2
	NET2_CKTP = args.net2_cktp

	NET3 = args.net3
	NET3_CKTP = args.net3_cktp

	BATCH_SIZE 			= args.batch_size
	METHOD 				= args.method

	BITRATE 			= args.br

	print_args(args)



	################################################## Setup the training options
	# Load model
	NUM_CLASS = 4 if METHOD == "multi" else 2
	model1 = load_model(NET1, 0.0, NUM_CLASS)
	model2 = load_model(NET2, 0.0, NUM_CLASS)
	model3 = load_model(NET3, 0.0, NUM_CLASS)


	# load the model weights
	load_ckpt(model1, NET1_CKTP, 0, True)
	load_ckpt(model2, NET2_CKTP, 0, True)
	load_ckpt(model3, NET3_CKTP, 0)
	


	################################################## Load the test files
	# Set test data
	test_fnames = txt2list(glob(join(SRC_PATH, METHOD, "test_*.txt")))
	test_fnames = list(filter(lambda x: BITRATE in x, test_fnames))[:10]
	dataset 	= configure_dataset(test_fnames, BATCH_SIZE, shuffle=False)

	

	################################################## Test the model
	total_length = 2 * len(test_fnames)
	STEPS_TEST = total_length // BATCH_SIZE

	input_data = []
	input_label = []
	'''
	inputs = Input(shape=(256, 256, 1))
	_, gap1 = model1(inputs)
	_, gap2 = model2(inputs)
	concat = tf.concat([gap1, gap2], axis=-1)
	out = model3(concat)

	model = Model(inputs=inputs, outputs=out)
	model.compile()
	model.summary()
	'''

	for frame, label in dataset:
		_, gpa1 = model1(frame)
		_, gpa2 = model2(frame)
		input_data = tf.concat([gpa1, gpa2], axis=-1)
		outt = model3(input_data)
		print(label.numpy())
		print(outt.numpy())




if __name__ == "__main__":
	main()


