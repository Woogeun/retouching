
import argparse
from glob import glob
from os import cpu_count
from os.path import join

import tensorflow as tf
from tensorflow import keras

from network.Networks_structure_srnet_keras import SRNet, SRNet_


def main():
	################################################## parse the arguments
	parser = argparse.ArgumentParser(description='Train retouch detection network.')
	parser.add_argument('--file_path', type=str, default='./testS_output/blur/mclv_S_0_1_500_0_.mp4', help='test file path')
	parser.add_argument('--method', type=str, default="blur", help='method')
	parser.add_argument('--network_scale', type=float, default=1.0, help='network scale')
	parser.add_argument('--batch_size', type=int, default=4, help='batch size')
	parser.add_argument('--checkpoint_path', type=str, default="./logs/20190816_100656_blur_95/cp-0007.ckpt", help='checkpoint path')
	parser.add_argument('--checkpoint_method', type=str, default="not", help='latest or not')
	args = parser.parse_args()

	FILE_PATH 			= args.file_path
	METHOD 				= args.method
	SCALE 				= args.network_scale
	BATCH_SIZE 			= args.batch_size
	CHECKPOINT_PATH 	= args.checkpoint_path
	CHECKPOINT_METHOD 	= args.checkpoint_method



	################################################## Import data
	sample = None
	


	################################################## Load checkpoint file
	model = SRNet()

	checkpoint = CHECKPOINT_PATH
	if CHECKPOINT_METHOD == "latest":
		checkpoint = tf.train.latest_checkpoint(CHECKPOINT_PATH)
	print(checkpoint)

	# model.load_weights(checkpoint)



	################################################## Predict the sample data
	metric_names = ["Loss", "Accuracy", "True Positive", "True Negative", "False Positive", "False Negative"]
	# result = model.predict(sample, verbose=1)
	# for key, value in zip(metric_names, result):
	# 	print("{:20s}: {:0.5f}\n".format(key, value))





















if __name__ == "__main__":
	main()


