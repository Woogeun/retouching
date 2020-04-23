from PIL import Image 
from glob import glob
from os.path import join, basename

methods = ["blur", "median", "noise"]



# for method in methods:
path = r"C:\Users\mmc\Desktop\spatial_result/"
frame_names = glob(path + "MVQP(*)_0*.png")
for frame_name in frame_names:
	frame = Image.open(frame_name)
	frame.crop((128,128,2048-128,1080-128)).save(path + f"cropped_{basename(frame_name)}")



# for method in methods:
# 	frame_names = glob(f"../frames_/strong/{method}/*.png")
# 	result_names = glob(f"../result_/strong/{method}/*.png")
# 	for frame_name, result_name in zip(frame_names, result_names):
# 		frame = Image.open(frame_name).convert('RGBA')
# 		result = Image.open(result_name).convert('RGBA')
# 		Image.blend(frame, result, 0.5).save(f"../frames_/strong/{method}/overlapped_{basename(frame_name)}")

