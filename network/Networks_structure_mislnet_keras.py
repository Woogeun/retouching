"""
MISLNet module
@authorized by Shasha Bae
@description: define the Bayar network(MISLNet)
"""

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.layers import Layer

from . import Networks_functions_keras
from .Networks_functions_keras import *




class ConstrainedLayer(Layer):
	"""Layer Class of first layer of Bayar network."""

	def __init__(self, filters, size, strides, **kwargs):
		super(ConstrainedLayer, self).__init__(**kwargs)
		constraint = CustomConstraint(sum=1, center=-1)
		self.conv = conv2D(filters, size, strides, kernel_constraint=constraint)

	def call(self, inputs):
		x = self.conv(inputs)
		return x



class MISLNet(Model):
	"""MISLNet class."""

	def __init__(self, scale=1.0, reg=0.001, **kwargs):
		super(MISLNet, self).__init__(**kwargs)
		Networks_functions_keras.SCALE = scale
		Networks_functions_keras.REG = reg

		# prediction error feature extraction
		self.conv1 			= ConstrainedLayer(3, (5,5), (1,1))


		# hierarchical feature extraction
		self.conv2 			= conv2D(96, (7,7), (2,2))
		self.batchNorm2 	= batchNorm()
		self.tanh2 			= tanh()
		self.maxPooling2 	= maxPooling2D((3,3), (2,2))

		self.conv3 			= conv2D(64, (5,5), (1,1))
		self.batchNorm3 	= batchNorm()
		self.tanh3 			= tanh()
		self.maxPooling3 	= maxPooling2D((3,3), (2,2))

		self.conv4 			= conv2D(64, (5,5), (1,1))
		self.batchNorm4 	= batchNorm()
		self.tanh4 			= tanh()
		self.maxPooling4 	= maxPooling2D((3,3), (2,2))


		# cross feature maps learning
		self.conv5 			= conv2D(128, (1,1), (1,1))
		self.batchNorm5 	= batchNorm()
		self.tanh5 			= tanh()
		self.averagePooling5 = averagePooling2D((3,3), (2,2))


		# classification
		self.flatten 	= flatten()
		self.fc1 		= dense(200, use_bias=False, activation='tanh')
		self.fc2 		= dense(200, use_bias=False, activation='tanh')
		self.fc3 		= dense(2, use_bias=False, activation='softmax')


	def call(self, inputs):
		x = self.conv1(inputs)
		
		x = self.conv2(x)
		x = self.batchNorm2(x)
		x = self.tanh2(x)
		x = self.maxPooling2(x)

		x = self.conv3(x)
		x = self.batchNorm3(x)
		x = self.tanh3(x)
		x = self.maxPooling3(x)

		x = self.conv4(x)
		x = self.batchNorm4(x)
		x = self.tanh4(x)
		x = self.maxPooling4(x)

		x = self.conv5(x)
		x = self.batchNorm5(x)
		x = self.tanh5(x)
		x = self.averagePooling5(x)

		x = self.flatten(x)
		x = self.fc1(x)
		x = self.fc2(x)
		x = self.fc3(x)

		return x











