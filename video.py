import argparse
import glob
import os
from manipulate import manipulate
import skvideo.io as vio # pip install sk-video
import numpy as np

def main():

	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('--path', type=str, default='./test', help='directory path')
	parser.add_argument('--method', type=str, default='all', help='blur, median, noise, or all')
	args = parser.parse_args()

	src_path = args.path
	if not os.path.isdir(src_path):
		raise(BaseException("no such directory \"%s\"" % src_path))

	method = args.method
	methods = ["blur", "median", "noise"]
	if (method not in methods) and (method != "all"):
		raise(BaseException("no such method \"%s\"" % method))


	dst_path = src_path + "_output0"
	idx = 1

	while os.path.isdir(dst_path):
		if idx > 9:
			raise(BaseException("so many directories"))
		dst_path = dst_path[:-1]
		dst_path += str(idx)
		idx += 1

	src_path += "/"
	dst_path += "/"

	os.mkdir(dst_path)
	for m in methods:
		os.mkdir(dst_path + m)

	


	counter = 1
	fnames = glob.glob(src_path + "*")
	print("%8s| file name" % "counter")

	for fname in fnames:
		meta = vio.ffprobe(fname)
		# for x in meta:
		# 	print(x)
		# 	for y in meta[x]:
		# 		print (y, ':', meta[x][y])
		# return
		vid = np.array(vio.vread(fname))
		vid_retouched = np.zeros(vid.shape)
		fn, w, h, c = vid.shape

		if method == "all":
			for m in methods:
				for i in range(fn):
				    vid_retouched[i,:,:,:] = manipulate(vid[i,:,:,:], m) # manipulate.py 참고

				vid_retouched = vid_retouched.astype(np.uint8)

				output_file = dst_path + m + "/" + os.path.basename(fname)
				print("%8d: %s" % (counter , output_file))
				counter += 1
				writer = vio.FFmpegWriter(filename=output_file, outputdict={'-r': '30', '-g': '4'}) # outputdict에 bitrate, gop 등 추가
				for i in range(fn):
				        writer.writeFrame(vid_retouched[i, :, :, :])
				writer.close()

		else:
			for i in range(fn):
			    vid_retouched[i,:,:,:] = manipulate(vid[i,:,:,:], method) # manipulate.py 참고

			vid_retouched = vid_retouched.astype(np.uint8)

			output_file = dst_path + method + "/" + os.path.basename(fname)
			print("%8d: %s" % (counter , output_file))
			counter += 1
			writer = vio.FFmpegWriter(filename=output_file, outputdict={}) # outputdict에 bitrate, gop 등 추가
			for i in range(fn):
			        writer.writeFrame(vid_retouched[i, :, :, :])
			writer.close() 

		



	print("Process end on directory \"%s\"" % dst_path)
	# meta = vio.ffprobe('mclv_S_0_1_500_0_.mp4')
	# vid = np.array(vio.vread('mclv_S_0_1_500_0_.mp4'))
	# fn, w, h, c = vid.shape

	# for i in range(fn):
	#     vid[i,:,:,:] = manipulate(vid[i,:,:,:], 'gaussNoise_2') # manipulate.py 참고

	# out = vid.astype(np.uint8)
	# writer = vio.FFmpegWriter(filename="out.mp4", outputdict={}) # outputdict에 bitrate, gop 등 추가
	# for i in range(fn):
	#         writer.writeFrame(out[i, :, :, :])
	# writer.close() 


if __name__=="__main__":
	main()