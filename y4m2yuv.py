import os
from os.path import join, basename
from glob import glob

dir = "E:\\YUV_RAWVIDEO\\XIPH2"
fnames = glob(join(dir, "*.y4m"))
for fname in fnames:
	os.system("ffmpeg -i {} -pix_fmt yuv420p {}".format(fname, fname.split(".")[0] + ".yuv"))
