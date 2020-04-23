import cv2
import argparse
import glob
from os import makedirs
from os.path import isdir, join, basename
from manipulate import manipulate
import skvideo.io as vio # pip install sk-video
import numpy as np
from PIL import Image


def main():

	# parse the arguments
	parser = argparse.ArgumentParser(description='Retouch video.')
	parser.add_argument('--src_path', type=str, default='../retouch_temporal_videos', help='source path')
	parser.add_argument('--dst_path', type=str, default='../', help='destination path')
	parser.add_argument('--intensity', type=str, default='strong', help='strong or weak')
	args = parser.parse_args()

	# source directory validation check
	src_path 	= args.src_path
	dst_path 	= args.dst_path
	intensity  	= args.intensity

	# method validation check
	methods = ["blur", "median", "noise"]

	# set destination directory name
	for method in methods:
		try:
			makedirs(join(dst_path, intensity, method))
			pass
		except FileExistsError:
			pass

	counter = 1
	fnames = glob.glob(join(src_path, intensity, "*.mp4"))
	print("%8s| file name" % "counter")

	for method in methods:
		fnames = glob.glob(join(src_path, intensity, method, "*.mp4"))
		for fname in fnames:
			vid = np.array(vio.vread(fname))
			vid_meta = vio.ffprobe(fname)
			# print(vid_meta)
			output_file = join(dst_path, intensity, method, fname.split("\\")[-1].split(".")[0] + ".png")
			img = Image.fromarray(vid[0], 'RGB')
			img.save(output_file)
			print("%8d: %s" % (counter , output_file))
			counter += 1
	
	print("Process end on directory \"%s\"" % src_path)


	
if __name__=="__main__":
	main()