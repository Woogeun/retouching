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

from utils import *




__DUMMY__ = 0.0001

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


def main():
	################################################## Parse the arguments
	parser = argparse.ArgumentParser(description='Train retouch detection network.')
	parser.add_argument('--src_path', type=str, 	default='./fusion/result', help='test source path')
	parser.add_argument('--dst_path', type=str, 	default='../spatial_localization', help='image save destination path')

	args = parser.parse_args()

	SRC_PATH 	= args.src_path
	DST_PATH 	= args.dst_path

	print_args(args)



	################################################## Load model with checkpoint
	fnames = glob(join(SRC_PATH, "*.npy"))
	makedirs(DST_PATH, exist_ok=True)

	for fname in fnames:
		f = np.load(fname)
		f_idx = np.argmax(f, axis=-1)
		height, width, _ = f.shape
		img_np = np.zeros((height, width, 3))

		if "blur" in fname: method = "blur"
		elif "median" in fname: method = "median"
		elif "noise" in fname: method = "noise"

		for h in range(height):
			for w in range(width):
				img_np[h, w] = setColor(f_idx[h, w], two_color=True, method=method)
				img = Image.fromarray(img_np.astype('uint8'), 'RGB')

		output_file = join(DST_PATH, basename(fname).split('.')[0] + '.png')
		img.save(output_file)
		print(output_file)




if __name__ == "__main__":
	main()


