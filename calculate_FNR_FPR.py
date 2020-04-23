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
	# parser.add_argument('--net_cktp', type=str, 	default="./logs/20191104_171914_blur_SRNet_98/checkpoint/weights_18", help='checkpoint path')
	# parser.add_argument('--net_cktp', type=str, 	default="./logs/20191107_134614_median_SRNet_98/checkpoint/weights_18", help='checkpoint path')
	parser.add_argument('--net_cktp', type=str, 	default="./logs/20191111_114343_noise_SRNet_90/checkpoint/weights_35", help='checkpoint path')
	# parser.add_argument('--net_cktp', type=str, 	default="./logs/20191014_142248_multi_SRNet_93/checkpoint/weights_18", help='checkpoint path')
	# parser.add_argument('--net_cktp', type=str, 	default="./logs/20200306_143112_blur_Total_98/checkpoint/weights_29", help='checkpoint path')
	# parser.add_argument('--net_cktp', type=str, 	default="./logs/20200309_143754_median_Total_98/checkpoint/weights_21", help='checkpoint path')
	# parser.add_argument('--net_cktp', type=str, 	default="./logs/20200322_030005_noise_Total_90/checkpoint/weights_11", help='checkpoint path')
	
	parser.add_argument('--batch_size', type=int, 	default=64, help='batch size')
	# parser.add_argument('--method', type=str, 		default="blur", help='blur median noise multi')
	# parser.add_argument('--method', type=str, 		default="median", help='blur median noise multi')
	parser.add_argument('--method', type=str, 		default="noise", help='blur median noise multi')
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
	


	################################################## Load the test files
	# Set test data
	dataset, total = configure_dataset(   SRC_PATH, METHOD, 'test', BATCH_SIZE, \
                                        shuffle=True, repeat=False)

	

	################################################## Test the model
	labels = np.zeros((total, 2))
	preds = np.zeros((total, 2))
	

	offset = 0
	for frames, labels_ in dataset:
		
		batch_size = labels_.shape[0]

		labels[offset:offset+batch_size, :] = labels_
		preds[offset:offset+batch_size, :] = model(frames)

		offset += batch_size

		progress = 100 * offset / total
		print("offset: {0:.2f}%".format(progress), end='\r' if progress < 100 else '\n')
		if progress < 100: sys.stdout.flush()

	labels = labels[:,1]
	preds = preds[:,1]
		

	results = []
	THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.6, 0.7, 0.8, 0.9]
	# for THRESHOLD in np.arange(0.1, 1.0, 0.1):
	for THRESHOLD in THRESHOLDS:
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
		

		# ROC = TPR / FPR
		results += [[THRESHOLD, ACC, FNR, FPR]]

	for result in results:
		print(f"THRESHOLD: {result[0]:.2f}, 정확도: {result[1] * 100:.2f}%, 미탐지율: {result[2] * 100:.2f}%, 오탐지율: {result[3] * 100:.2f}%")




if __name__ == "__main__":
	main()


