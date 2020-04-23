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




BLACK = (0,0,0)
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)

colors = [BLACK, RED, GREEN, BLUE]

def setColor(result, two_color=True, method=None):
	# assert(two_color and method)
	if two_color:
		if result == 1 and method == "blur": color = colors[result]
		elif result == 2 and method == "median": color = colors[result]
		elif result == 3 and method == "noise": color = colors[result]
		else: color = colors[0]

	else:
		color = colors[result]

	return color


__DUMMY__ = 0.0001
def main():
	################################################## Parse the arguments
	parser = argparse.ArgumentParser(description='Train retouch detection network.')
	parser.add_argument('--src_path', type=str, 	default='../frames', help='test source path')
	parser.add_argument('--dst_path', type=str, 	default='./fusion/result', help='image save destination path')

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
		fnames = glob(join(SRC_PATH, INTENSITY, method, "*.png"))
		for fname in fnames:
			frame = np.array(Image.open(fname))
			frame_gray = np.array(Image.open(fname).convert('L'))
			h, w = frame_gray.shape
			frame_gray = np.reshape(frame_gray, (h, w ,1))

			predicted_data = detector.detect_frame_full(frame_gray, h, w)
			output_file = join(DST_PATH, method, "{}_{}_{}".format(fname.split("\\")[-1].split('.')[0], method, INTENSITY))

			np.save(output_file, predicted_data)

			f_idx = np.argmax(predicted_data, axis=-1)
			height, width = f_idx.shape
			img_np = np.zeros((height, width, 3))



			for h in range(height):
				for w in range(width):
					img_np[h, w] = setColor(f_idx[h, w], two_color=True, method=method)
					# img_np[h, w] = setColor(f_idx[h, w], two_color=False)

			img = Image.fromarray(img_np.astype('uint8'), 'RGB')
			
			result = img.resize((2048-256,1080-256)).convert('RGBA')
			result.save(join(DST_PATH, method, f"{basename(fname).split('.')[0]}_{INTENSITY}_result.png"))
			frame = Image.open(fname).crop((128,128,2048-128,1080-128)).convert('RGBA')
			output = Image.blend(frame, result, 0.5)

			output_file = join(DST_PATH, method, f"{basename(fname).split('.')[0]}_{INTENSITY}_overlapped.png")
			output.save(output_file)
			print(output_file)





if __name__ == "__main__":
	main()


