"""
Network predictor module
@authorized by Shasha Bae
@description: predicts input video tampering
"""

from os import makedirs
from os.path import join, basename

import numpy as np
import skvideo.io as vio
import matplotlib.pyplot as plt

import tensorflow as tf

from utils import *




class Detector():
	"""Class for the detecting whether the video is tampered or not."""
	def __init__(self, model1, model2=None, model3=None, dst_path=None):
		self.model1 = model1
		self.model2 = model2
		self.model3 = model3
		self.dst_path = dst_path
		self.is_DPN = True if model2 else False

		makedirs(dst_path, exist_ok=True)




	def detect_frame(self, frame):
		"""Detect a framewise tampering

	   	# arguments
	   		frame: 256x256 numpy array of one frame

		# Returns
			The result accuracy of the frame can be tampered.
		"""

		if self.is_DPN:
			_, gpa1 = self.model1([frame])
			_, gpa2 = self.model2([frame])
			input_gpa = tf.concat([gpa1, gpa2], axis=-1)

			result_accuracy = self.model3(np.array(input_gpa))
		else:
			frame = np.reshape(frame, [1, 256, 256, 1]).astype('float32')
			result_accuracy = self.model1(frame)

		return result_accuracy.numpy()


	def detect_frame_full(self, frame, width, height):
		"""Detect a full size framewise tampering 

	   	# arguments
	   		frame: numpy array of one frame
	   		width: width of frame
	   		height: height of frame

		# Returns
			The result accuracy of the frame can be tampered.
		"""
		strides = 16
		_width = int((width - 256 + strides) / strides)
		_height = int((height - 256 + strides) / strides)

		num_outs = _width * _height
		batch_size = 100
		num_batchs = num_outs // batch_size

		result = np.zeros((_width, _height, 4))
		inputs = np.zeros((_height, 256, 256, 1))

		

		if self.is_DPN:
			for w in range(0, _width):
				idx = 0

				for h in range(0, _height):
					print_progress("progress: {:4.4f}%", 100 * (w * _height + h) / _width / _height)
					inputs[idx,:,:,:] = frame[w * strides:w * strides + 256, h * strides:h * strides + 256, :]
					idx += 1
					
				_, gpa1 = self.model1(inputs)
				_, gpa2 = self.model2(inputs)
				input_gpa = tf.concat([gpa1, gpa2], axis=-1)

				result[w,:,:] = self.model3(input_gpa)

			

		else:
			for w in range(0, width, strides):
				for h in range(0, height, strides):
					print_progress("progress: {:4.4f}", 100 * progress / width / height)

					window = frame[w:w+256, h:h+256, :]
					accuracy = self.model1(np.array([window]))
					result[w//8, h//8, :] = accuracy

					progress += strides


		return result


	def is_original(self, data):
		"""Determine the videowise tampering

	   	# arguments
	   		data: the list of the probability of frame can be tampered

		# Returns
			The bool that input video is not tampered. If the video is not tampered, returns True, else False.
		"""
		
		return None


	def detect_video(self, video_name, method, intensity):
		"""Detect a videowise tampering

	   	# arguments
	   		video_name: the string of video file name
	   		is_show: the bool of whether show the result graph

		# Returns
			The tuple of whether the video is tampered and predicted data
		"""

		# Read video
		video = np.array(vio.vread(video_name, outputdict={"-pix_fmt": "gray"})).astype('float32')
		fn, w, h, c = video.shape
		if w != 256 or h != 256 or c != 1: 
			raise(BaseException("================ wrong size file: \"{}\"".format(fname)))


		# Predict the video retouch tampering
		predicted_data = []
		for idx in range(fn):
			result = self.detect_frame(video[idx])
			predicted_data.append(result)

		predicted_data = np.array(predicted_data)

		predicted_data = np.reshape(predicted_data, (fn, 4))
			

		# Determine the prediction result(original or not) and save figure
		prediction = self.is_original(predicted_data)
		
		plt.plot(range(fn), predicted_data[:,0], 'k--', \
				range(fn), predicted_data[:,1], 'rs', range(fn), \
				predicted_data[:,2], 'g^', range(fn), \
				predicted_data[:,3], 'bo')
		output_file = join(self.dst_path, intensity, method, f"{basename(video_name).split('.')[0] + '.png'}")
		plt.savefig(output_file)
		plt.clf()

		return predicted_data

