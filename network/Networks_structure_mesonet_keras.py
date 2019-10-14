"""
MesoNet module
@authorized by Shasha Bae
@description: define the MesoNet
"""

import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import Model, Input

from . import Networks_functions_keras
from .Networks_functions_keras import *


class ConvBlock(Layer):
	"""Conv, batchNorm, max pooling block class."""
	def __init__(self, filters, kernel_size, pool_size, **kwargs):
		super(ConvBlock, self).__init__(**kwargs)

		self.conv = conv2D(filters, kernel_size)
		self.batchNorm = batchNorm()
		self.maxPooling = maxPooling2D(pool_size)

	def __call__(self, inputs):
		x = self.conv(inputs)
		x = self.batchNorm(x)
		x = self.maxPooling(x)

		return x


class DropoutBlock(Layer):
	"""Dropout and fully connected class."""
	def __init__(self, fc, **kwargs):
		super(DropoutBlock, self).__init__(**kwargs)

		self.dropout = dropout(0.5)
		self.fc = dense(fc)

	def __call__(self, inputs):
		x = self.dropout(inputs)
		x = self.fc(x)

		return x


class MesoNet(Model):
	"""MesoNet class."""

	def __init__(self, scale=1.0, reg=0.001, num_class=2, **kwargs):
		super(MesoNet, self).__init__(**kwargs)
		Networks_functions_keras.SCALE = scale
		Networks_functions_keras.REG = reg

		# convolutional blocks
		self.block1 = ConvBlock(8, (3,3), (2,2))
		self.block2 = ConvBlock(8, (5,5), (2,2))
		self.block3 = ConvBlock(16, (5,5), (2,2))
		self.block4 = ConvBlock(16, (5,5), (4,4))


		# dropout blocks
		self.dropout1 = DropoutBlock(16)
		self.dropout2 = DropoutBlock(num_class)
		

		# softmax
		self.softmax = softmax()
		

	def call(self, inputs):
		x = self.block1(inputs)
		x = self.block2(x)
		x = self.block3(x)
		x = self.block4(x)

		x = self.dropout1(x)
		x = self.dropout2(x)

		x = self.softmax(x)

		return x













