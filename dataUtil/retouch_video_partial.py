
import skvideo.io as vio
import numpy as np
from manipulate import manipulate
import argparse
from glob import glob
from os import makedirs
from os.path import join
import random



fps = '30'
write_option = {'-vcodec': 'libx264', '-r': fps, '-g': '4', '-bf': '0', '-pix_fmt': 'yuv420p'}
# write_option = {}


def rename(fname, method, DST_PATH):
	return join(DST_PATH, fname.split('\\')[-1].split('.')[0] + "_{}_partial.mp4".format(method))

def partial_retouch(fname):
	pass


def main():
	# parse the arguments
	parser = argparse.ArgumentParser(description='Retouch video partially.')
	parser.add_argument('--src_path', type=str, default='./4k_videos_split', help='source path')
	parser.add_argument('--dst_path', type=str, default='./4k_video_temporal_localization', help='destination path')
	args = parser.parse_args()

	SRC_PATH 	= args.src_path
	DST_PATH 	= args.dst_path

	methods = ["blur", "median", "noise"]
	intensities = ["strong", "weak"]
	bitrates = ["500k", "600k", "700k", "800k"]

	for intensity in intensities:
		for method in methods:
			for bitrate in bitrates:
				makedirs(join(DST_PATH, intensity, method, bitrate), exist_ok=True)


	fnames = glob(join(SRC_PATH, "*k", "*.mp4"))

	for fname in fnames:
		vOriginal = np.array(vio.vread(fname, inputdict={}))
		vRetouched = np.zeros(vOriginal.shape)

		fn, w, h, c = vOriginal.shape

		start_fn = int(fn / 3)
		end_fn = int(fn*2 / 3)

		for intensity in intensities:
			for method in methods:
				for bitrate in bitrates:
					
					print("processing {}".format(join(DST_PATH, intensity, method, bitrate, fname.split("\\")[-1])))

					for idx in range(fn):
						if idx >= start_fn and idx <= end_fn:
							vRetouched[idx] = manipulate(vOriginal[idx], method, intensity=intensity)
						else:
							vRetouched[idx] = vOriginal[idx]

					write_option['-b:v'] = bitrate
					writer = vio.FFmpegWriter(filename=join(DST_PATH, intensity, method, bitrate, fname.split("\\")[-1]), inputdict={'-r':fps}, outputdict=write_option)
					for frame in vRetouched:
						writer.writeFrame(frame)
					writer.close()

					return




if __name__ == "__main__":
	main()


