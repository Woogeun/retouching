"""
SPN prediction module
@authorized by Shasha Bae
@description: predict whether the input video is tampered or not. If input type is directory, predict the videos in the directory 
"""

import argparse
from os import makedirs
from os.path import join, isdir, basename
from PIL import Image
from glob import glob

import tensorflow as tf
from tensorflow import keras

from utils import *
from network import *
from predictor import Detector




__DUMMY__ = 0.0001
def main():
	################################################## Parse the arguments
	parser = argparse.ArgumentParser(description='Train retouch detection network.')
	parser.add_argument('--src_path', type=str, 	default=r'E:\YUV_RAWVIDEO\4k_video_temporal_localization', help='test source path')
	parser.add_argument('--dst_path', type=str, 	default='../temporal_result', help='image save destination path')

	parser.add_argument('--method', type=str, 				default="multi", help='blur, median, noise or multi')
	parser.add_argument('--regularizer', type=float, 		default=0.0001, help='regularizer')
	parser.add_argument('--start_lr', type=float, 			default=1e-05, help='start learning rate')

	parser.add_argument('--network1', type=str, default="SRNet", help='SRNet or MISLNet or DCTNet or MesoNet')
	parser.add_argument('--checkpoint1_path', type=str, default="./logs/20191014_142248_multi_SRNet_93/checkpoint/weights_10", help='checkpoint path')

	parser.add_argument('--network2', type=str, default="DCTNet", help='SRNet or MISLNet or NamNet or MMCNet or DCTNet or MesoNet')
	parser.add_argument('--checkpoint2_path', type=str, default="./logs/20191016_083522_multi_DCTNet_88/checkpoint/weights_14", help='checkpoint path')

	parser.add_argument('--network3', type=str, 			default="Fusion", help='SRNet or MISLNet or DCTNet or MesoNet')
	parser.add_argument('--checkpoint3_path', type=str, 	default="./logs/20191203_140056_multi_Fusion_96/checkpoint/weights_10", help='checkpoint3 path')

	parser.add_argument('--intensity', type=str, 	default="strong", help='strong or weak')
	parser.add_argument('--bitrate', type=str, default="800k", help='500k, 600k, 700k, or 800k')
	
	args = parser.parse_args()

	SRC_PATH 	= args.src_path
	DST_PATH 	= args.dst_path

	METHOD 		= args.method
	REG 		= args.regularizer
	START_LR 	= args.start_lr

	NETWORK1 			= args.network1
	CHECKPOINT1_PATH 	= args.checkpoint1_path

	NETWORK2 			= args.network2
	CHECKPOINT2_PATH 	= args.checkpoint2_path

	NETWORK3 			= args.network3
	CHECKPOINT3_PATH 	= args.checkpoint3_path

	INTENSITY 	= args.intensity
	BITRATE  	= args.bitrate

	print_args(args)



	################################################## Load model with checkpoint
	NUM_CLASS = 4 if METHOD == "multi" else 2
	
	model1 = load_model(NETWORK1, REG, NUM_CLASS)
	model2 = load_model(NETWORK2, REG, NUM_CLASS)
	model3 = load_model(NETWORK3, REG, NUM_CLASS)
	
	load_ckpt(model1, CHECKPOINT1_PATH, START_LR)
	load_ckpt(model2, CHECKPOINT2_PATH, START_LR)
	load_ckpt(model3, CHECKPOINT3_PATH, START_LR)


	################################################## Predict the data
	detector = Detector(model1=model1, model2=model2, model3=model3, dst_path=DST_PATH)

	methods = ["blur", "median", "noise"]

	for method in methods:
		fnames = glob(join(SRC_PATH, INTENSITY, method, BITRATE, "*.mp4"))[:100]
		for fname in fnames:
			predicted_data = detector.detect_video(fname, method, INTENSITY)
			output_file = join(DST_PATH, INTENSITY, method, BITRATE, "{}_{}".format(fname.split("\\")[-1].split('.')[0], method))
			np.save(output_file, predicted_data)

			print(output_file)





if __name__ == "__main__":
	main()


