"""
MISLNet module
@authorized by Shasha Bae
@description: define the Bayar network(MISLNet)
"""

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.layers import Layer

# from . import Networks_functions
from .Networks_functions import *


class MesoNet(Model):
	"""MESONet class."""

	def __init__(self, reg=0.001, num_class=2, **kwargs):
		super(MesoNet, self).__init__(**kwargs)
		set_parameters(reg, num_class)

		self.conv1 = conv2D(8, (3,3), (1,1))
		self.relu1 = relu()
		self.batchNorm1 = batchNorm()
		self.maxPooling2D1 = maxPooling2D((2,2),(2,2))

		self.conv2 = conv2D(8, (5,5), (1,1))
		self.relu2 = relu()
		self.batchNorm2 = batchNorm()
		self.maxPooling2D2 = maxPooling2D((2,2),(2,2))

		self.conv3 = conv2D(16, (5,5), (1,1))
		self.relu3 = relu()
		self.batchNorm3 = batchNorm()
		self.maxPooling2D3 = maxPooling2D((2,2),(2,2))

		self.conv4 = conv2D(16, (5,5), (1,1))
		self.relu4 = relu()
		self.batchNorm4 = batchNorm()
		self.maxPooling2D4 = maxPooling2D((4,4),(4,4))

		self.flt = flatten()
		self.fc1 = dense(16, use_bias=False, activation=None)
		self.fc2 = dense(num_class, use_bias=False, activation='softmax')

	def call(self, inputs):
		x = self.conv1(inputs)
		x = self.relu1(x)
		x = self.batchNorm1(x)
		x = self.maxPooling2D1(x)

		x = self.conv2(x)
		x = self.relu2(x)
		x = self.batchNorm2(x)
		x = self.maxPooling2D2(x)

		x = self.conv3(x)
		x = self.relu3(x)
		x = self.batchNorm3(x)
		x = self.maxPooling2D3(x)

		x = self.conv4(x)
		x = self.relu4(x)
		x = self.batchNorm4(x)
		x = self.maxPooling2D4(x)		

		x = self.flt(x)
		x = self.fc1(x)
		x = self.fc2(x)

		return x











