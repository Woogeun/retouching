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

	parser.add_argument('--net', type=str, 			default="SRNet", help='SRNet or MISLNet or DCTNet or MesoNet')
	parser.add_argument('--net_scale', type=float, 	default=1.0, help='network scale')
	parser.add_argument('--net_cktp', type=str, 	default="./logs/20191104_171914_blur_SRNet_98/checkpoint/weights_18", help='checkpoint path')
	# parser.add_argument('--net_cktp', type=str, 	default="./logs/20191107_134614_median_SRNet_98/checkpoint/weights_18", help='checkpoint path')
	# parser.add_argument('--net_cktp', type=str, 	default="./logs/20191111_114343_noise_SRNet_90/checkpoint/weights_18", help='checkpoint path')
	# parser.add_argument('--net_cktp', type=str, 	default="./logs/20191014_142248_multi_SRNet_93/checkpoint/weights_18", help='checkpoint path')
	
	parser.add_argument('--batch_size', type=int, 	default=256, help='batch size')
	parser.add_argument('--method', type=str, 		default="blur", help='blur median noise multi')
	# parser.add_argument('--method', type=str, 		default="median", help='blur median noise multi')
	# parser.add_argument('--method', type=str, 		default="noise", help='blur median noise multi')
	# parser.add_argument('--method', type=str, 		default="multi", help='blur median noise multi')

	args = parser.parse_args()

	SRC_PATH 			= args.src_path

	NET 				= args.net
	NET_SCALE			= args.net_scale
	NET_CKTP 			= args.net_cktp

	BATCH_SIZE 			= args.batch_size
	METHOD 				= args.method

	print_args(args)



	################################################## Setup the training options
	# Load model
	NUM_CLASS = 4 if METHOD == "multi" else 2
	model = load_model(NET, NET_SCALE, __DUMMY__, NUM_CLASS)


	# load the model weights
	load_cktp(model, NET_CKTP, __DUMMY__)
	


	################################################## Load the test files
	# Set test data
	test_fnames = txt2list(glob(join(SRC_PATH, METHOD, "test_*.txt")))
	# test_fnames = list(filter(lambda x: "800k" in x, test_fnames))
	dataset 	= configure_dataset(test_fnames, BATCH_SIZE, shuffle=False)

	

	################################################## Test the model
	total_length = 2 * len(test_fnames)
	STEPS_TEST = total_length // BATCH_SIZE

	labels = np.zeros((total_length, 2))
	preds = np.zeros((total_length, 2))
	

	offset = 0
	for frames, labels_ in dataset:
		print("offset: {0:.2f}%".format(100 * offset / total_length))
		batch_size = labels_.shape[0]

		labels[offset:offset+batch_size, :] = labels_
		preds[offset:offset+batch_size, :] = model.predict(frames, verbose=1)

		offset += batch_size

	labels = labels[:,1]
	preds = preds[:,1]
		

	results = []
	for THRESHOLD in np.arange(0.1, 1.0, 0.1):
		preds_ = preds > THRESHOLD

		TP = 0
		TN = 0
		FP = 0
		FN = 0
		
		for label, pred in zip(labels, preds_):
			if 		label == 0 and pred == 0: TN += 1
			elif 	label == 0 and pred == 1: FP += 1
			elif 	label == 1 and pred == 0: FN += 1
			elif 	label == 1 and pred == 1: TP += 1

		print(f"TN: {TN} FP: {FP}, FN: {FN}, TP: {TP}")

		TPR = TP / (TP + FN)
		FNR = FN / (TP + FN) # 미탐지율
		FPR = FP / (FP + TN) # 오탐지율
		ACC = (TP + TN) / (TP + TN + FP + FN)
		

		ROC = TPR / FPR
		results += [[THRESHOLD, ACC, FNR, FPR]]

	for result in results:
		print(f"THRESHOLD: {result[0]:.1f}, 정확도: {result[1] * 100:.2f}%, 미탐지율: {result[2] * 100:.2f}%, 오탐지율: {result[3] * 100:.2f}%")




if __name__ == "__main__":
	main()


