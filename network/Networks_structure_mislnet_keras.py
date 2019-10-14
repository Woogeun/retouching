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


class ConvBlock(Layer):
	"""Conv, batchNorm, max pooling block class."""
	def __init__(self, filters, kernel_size, strides, **kwargs):
		super(ConvBlock, self).__init__(**kwargs)
		self.conv = conv2D(filters, kernel_size, strides)
		self.batchNorm = batchNorm()
		self.tanh = tanh()
		self.maxPooling = maxPooling2D((3,3), (2,2))

	def __call__(self, inputs):
		x = self.conv(inputs)
		x = self.batchNorm(x)
		x = self.tanh(x)
		x = self.maxPooling(x)

		return x


class MISLNet(Model):
	"""MISLNet class."""

	def __init__(self, scale=1.0, reg=0.001, num_class=2, **kwargs):
		super(MISLNet, self).__init__(**kwargs)
		Networks_functions_keras.SCALE = scale
		Networks_functions_keras.REG = reg

		# prediction error feature extraction
		self.conv 	= ConstrainedLayer(3, (5,5), (1,1))


		# hierarchical feature extraction
		self.block1 = ConvBlock(96, (7,7), (2,2))
		self.block2 = ConvBlock(64, (5,5), (2,2))
		self.block3 = ConvBlock(64, (5,5), (2,2))
		self.block4 = ConvBlock(128, (1,1), (1,1))


		# classification
		self.flatten 	= flatten()
		self.fc1 		= dense(200, use_bias=False, activation='tanh')
		self.fc2 		= dense(200, use_bias=False, activation='tanh')
		self.fc3 		= dense(num_class, use_bias=False, activation='softmax')


	def call(self, inputs):
		x = self.conv(inputs)
		
		x = self.block1(x)
		x = self.block2(x)
		x = self.block3(x)
		x = self.block4(x)

		x = self.flatten(x)
		x = self.fc1(x)
		x = self.fc2(x)
		x = self.fc3(x)

		return x











