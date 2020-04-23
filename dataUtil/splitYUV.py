import numpy as np
import cv2
import skvideo.io as vio
import mmap

from os import makedirs
from os.path import join, basename
import argparse
from glob import glob




# load writer with parameter
# "-vcodec = libx264" 	: h.264 codec
# "-r = 30" 			: fps
# "-g = 4"				: GOP size
# "-bf = 0" 			: number of b frame
# "-b:v = bitrate" 		: bitrate
# "-pix_fmt = yuv420p"	: color space
fps = '30'
write_option = {'-vcodec': 'libx264', '-r': fps, '-g': '4', '-bf': '0', '-pix_fmt': 'yuv420p'}


def basename_without_extension(fname):
	return basename(fname).split(".")[0]


class YUVReader():
	'''Class for read YUV video and write video as mp4 format with 256x256 size'''
	def __init__(self, num_frames=200):
		"""Configure the video type and maximum number of frames to be read

	   	# arguments
	   		vtype: "mvqp", "mclv", or "xiph2"
	   		num_frames: maxinum number of frame to be read. It`s ok that input video has shorter frame than num_frames.
		"""

		self.num_frames = num_frames

		
	def clip_frame(self, frame, height, width, flag=True):
		if flag:
			cliped = []
			for h in range(0, height, 256):
				for w in range(0, width, 256):
					cliped_frame = frame[h : h+256, w : w+256, :]
					if cliped_frame.shape == (256,256,3):
						cliped.append(cliped_frame)

			return len(cliped), np.array(cliped)

		else:
			return 1, np.array([frame])



	def parseVideo(self, vname, vtype, dst_path):
		"""parse Video and write as a mp4 file

	   	# arguments
	   		vname: input video name
	   		dst_path: destination directory path, not file name
		"""

		# Data format of cliped_Frames is
		# [num_frames, num_clips, 256, 256, 3]
		if vtype == "MVQP":
			height = 1080
			width = 2048
		elif vtype == "MCLV":
			height = 1080
			width = 1920
		elif vtype == "XIPH2":
			height = 1080
			width = 1920
		elif vtype == "MCML":
			height = 2160
			width = 3840

		cliped_Frames = []
		num_clips = 0
		print("***********: ", vname)


		##################################### Store the cliped video as numpy array in self.cliped_Frames
		with  open(vname, 'rb') as stream:
			m = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)

			for idx in range(self.num_frames):
				try:	# if number of video frame is less than self.num_frames, it stops reading video
					YUV =  np.zeros([height, width, 3], dtype=np.uint8)
					YUV[:, :, 0] = np.frombuffer(m, dtype=np.uint8, count=width * height, offset = int(idx * width * height * 1.5)).reshape(height, width)
					YUV[:, :, 1] = np.frombuffer(m, dtype=np.uint8, count=(width // 2) * (height // 2), offset = int(idx * width * height * 1.5 + width * height )).reshape(height // 2, width // 2).repeat(2, axis=0).repeat(2, axis=1)
					YUV[:, :, 2] = np.frombuffer(m, dtype=np.uint8, count=(width // 2) * (height // 2), offset = int(idx * width * height * 1.5 + width * height * 1.25 )).reshape(height // 2, width // 2).repeat(2, axis=0).repeat(2, axis=1)
					RGB = cv2.cvtColor(YUV, cv2.COLOR_YUV2RGB)
				except Exception as e:
					print(e)
					break


				# Divide large size frame into 256x256 size video
				num_clips, clips = self.clip_frame(RGB, height, width, flag=False)
				cliped_Frames.append(clips)



		###################################### Write the numpy array as mp4 file
		cliped_Frames = np.array(cliped_Frames)
		# bitrates = ["500k", "600k", "700k", "800k"]
		bitrates = ["27M"]

		
		for clip_idx in range(num_clips):
			clip = cliped_Frames[:, clip_idx]
		
			for bitrate in bitrates:
				write_option['-b:v'] = bitrate
				dir_path = join(dst_path, bitrate)
				try:
					makedirs(dir_path)
				except:
					pass
				output_fname = join(dir_path, "{}_{}.mp4".format(basename_without_extension(vname), str(clip_idx)))
				writer = vio.FFmpegWriter(filename=output_fname, inputdict={'-r': fps, '-pix_fmt': 'rgb24'}, outputdict=write_option)
		
				for frame in clip:
					writer.writeFrame(frame)

				writer.close()



def main():


	# # parse the arguments
	parser = argparse.ArgumentParser(description='YUV video to mp4 video')
	parser.add_argument('--src_path', type=str, default='E:\\YUV_RAWVIDEO', help='source path')
	parser.add_argument('--dst_path', type=str, default='../', help='destination path')
	args = parser.parse_args()

	src_path 	= args.src_path
	dst_path 	= args.dst_path

	yuvReader = YUVReader()
	# vtypes = ["MCLV", "MVQP", "XIPH2"]
	# vtypes = ["XIPH2"]
	# vtypes = ["MCLV"]
	vtypes = ["MVQP"]
	# vtypes = ["MCML"]


	for vtype in vtypes:
		fnames = glob(join(src_path, vtype, "*.yuv"))
		fnames += glob(join(src_path, vtype, "*.YUV"))
		for fname in fnames:
			yuvReader.parseVideo(fname, vtype, dst_path)


	
#############################################

"""
	# Double compression
	vOriginal = np.array(vio.vread('./original/500k/BigBuckBunny_25fps_0.mp4'))
	vDouble = np.zeros(vOriginal.shape)
	for idx, frame in enumerate(vOriginal):
		vDouble[idx] = frame



	writer = vio.FFmpegWriter(filename="./original/500k/BigBuckBunny_25fps_0_double.mp4", inputdict={'-r': fps}, outputdict=write_option)
	for frame in vDouble:
		writer.writeFrame(frame)
	writer.close()



	# Frame number test
	meta_original = vio.ffprobe('./original/500k/BigBuckBunny_25fps_0.mp4')
	meta_double = vio.ffprobe('./original/500k/BigBuckBunny_25fps_0_double.mp4')

	for x in meta_original:
		print(x)
		for y in meta_original[x]:
			print (y, ':', meta_original[x][y])

	for x in meta_double:
		print(x)
		for y in meta_double[x]:
			print (y, ':', meta_double[x][y])

"""



if __name__=="__main__":
	main()

