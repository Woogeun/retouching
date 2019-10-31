"""
Network predictor module
@authorized by Shasha Bae
@description: predicts input video tampering
"""

from os.path import makedirs

import numpy as np
import skvideo.io as vio
import matplotlib.pyplot as plt

import tensorflow as tf




class Detector():
	"""Class for the detecting whether the video is tampered or not."""
	def __init__(self, model1, model2=None, model3=None, dst_path):
		self.model1 = model1
		self.model2 = model2
		self.model3 = model3
		self.dst_path = dst_path
		self.is_DPN = not model2

		makedirs(dst_path, exists_ok=True)




	def detect_frame(self, frame):
		"""Detect a framewise tampering

	   	# arguments
	   		frame: numpy array of one frame

		# Returns
			The result accuracy of the frame can be tampered. If frame is not tampered, result is closer to [1 0], else [0 1]
		"""

		if self.is_DPN:
			_, gpa1 = self.model1([frame])
			_, gpa2 = self.model2([frame])
			input_gpa = tf.concat([gpa1, gpa2], axis=-1)

			result_accuracy = self.model3.predict(np.array(input_gpa), verbose=0)
		else:
			result_accuracy = self.model1.predict(np.array([frame]), verbose=0)

		return result_accuracy


	def is_original(self, data):
		"""Determine the videowise tampering

	   	# arguments
	   		data: the list of the probability of frame can be tampered

		# Returns
			The bool that input video is not tampered. If the video is not tampered, returns True, else False.
		"""
		
		return None


	def detect_video(self, video_name, is_show=False):
		"""Detect a videowise tampering

	   	# arguments
	   		video_name: the string of video file name
	   		is_show: the bool of whether show the result graph

		# Returns
			The tuple of whether the video is tampered and predicted data
		"""

		# Read video
		video = np.array(vio.vread(video_name, outputdict={"-pix_fmt": "gray"}))
		fn, w, h, c = video.shape
		if w != 256 or h != 256 or c != 1: 
			raise(BaseException("================ wrong size file: \"{}\"".format(fname)))


		# Predict the video retouch tampering
		predicted_data = []
		for idx, frame in enumerate(video):
			result = self.detect_frame(frame)
			predicted_data.append(result)

			if is_show:
				print("{}: {}".format(idx, result))
			

		# Determine the prediction result(original or not) and save figure
		prediction = self.is_original(predicted_data)
		plt.plot(fn, predicted_data)
		

		if is_show:
			plt.show()
			plt.savefig(join(self.dst_path, video_name.split('\\')[-1].split('.')[0] + 'png'))
			print("Original: ", prediction)

		plt.clf()

		return predicted_data

