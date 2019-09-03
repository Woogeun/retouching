
import argparse
import glob
from os import makedirs
from os.path import isdir, join, basename
from manipulate import manipulate
import skvideo.io as vio # pip install sk-video
import numpy as np


def main():

	# parse the arguments
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('--src_path', type=str, default='./trainS_input', help='directory path')
	parser.add_argument('--frac', type=float, default='1.0', help='attack frac')
	args = parser.parse_args()

	# source directory validation check
	src_path 	= args.src_path
	frac  		= args.frac
	original_path = join(src_path, "original")
	if not isdir(original_path):
		raise(BaseException("no such directory \"%s\"" % original_path))

	# method validation check
	methods = ["blur", "median", "noise", "resize"]
	# methods = ["resize", "noise"]

	# set destination directory name
	for method in methods:
		try:
			makedirs(join(src_path, method))
			pass
		except FileExistsError:
			pass

	counter = 1
	fnames = glob.glob(join(original_path, "*.mp4"))
	print("%8s| file name" % "counter")

	# retouch video
	for fname in fnames:
		# video read
		meta = vio.ffprobe(fname)
		vid = np.array(vio.vread(fname))
		vid_retouched = np.zeros(vid.shape)
		fn, w, h, c = vid.shape
		if w != 256 or h != 256 or c != 3: 
			print("================ wrong size file: \"{}\"".format(fname))
			continue

		# parse bitrate from file name
		bitrate = basename(fname).split("_")[4] + "k"

		for method in methods:
			# get manipulated frame 
			for i in range(fn):
			    vid_retouched[i,:,:,:] = manipulate(vid[i,:,:,:], method, k=5, frac=frac) # manipulate.py 참고

			vid_retouched = vid_retouched.astype(np.uint8)

			# set output file name
			output_file = join(src_path, method, basename(fname))
			print("%8d: %s" % (counter , output_file))
			counter += 1
			
			# load writer with parameter
			# "-vcodec = libx264" 	: h.264 codec
			# "-r = 30" 			: fps
			# "-g = 4"				: GOP size
			# "-bf = 0" 			: number of b frame
			# "-b:v = bitrate" 		: bitrate
			# "-pix_fmt = yuv420p"	: color space
			write_option = {'-vcodec': 'libx264', '-r': '30', '-g': '4', '-bf': '0', '-b:v': bitrate, '-pix_fmt': 'yuv420p'}
			writer = vio.FFmpegWriter(filename=output_file, outputdict=write_option)
			for i in range(fn):
				writer.writeFrame(vid_retouched[i, :, :, :])
			writer.close()


	# for debug
	if False:
		print(basename(output_file))

		input_file = join(src_path + basename(output_file))
		input_meta = vio.ffprobe(input_file)
		for x in input_meta:
			print(x)
			for y in input_meta[x]:
				print (y, ':', input_meta[x][y])
		
		output_meta = vio.ffprobe(output_file)
		for x in output_meta:
			print(x)
			for y in output_meta[x]:
				print (y, ':', output_meta[x][y])


	print("Process end on directory \"%s\"" % src_path)

	
if __name__=="__main__":
	main()