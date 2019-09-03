"""
NamNet module
@authorized by Shasha Bae
@description: define the NamNet
"""

import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import Model, Input

from . import Networks_functions_keras
from .Networks_functions_keras import *




class Type1(Layer):
	"""Layer class of NamNet type 1."""

	def __init__(self, filters, **kwargs):
		super(Type1, self).__init__(**kwargs)

		self.conv1 = conv2D(filters, (3,3), (1,1))
		self.batchNorm1 = batchNorm()
		self.relu1 = relu()

		self.conv2 = conv2D(filters, (1,1), (1,1))
		self.batchNorm2 = batchNorm()
		self.relu2 = relu()

	def call(self, inputs):
		x = self.conv1(inputs)
		x = self.batchNorm1(x)
		x = self.relu1(x)

		x = self.conv2(inputs)
		x = self.batchNorm2(x)
		x = self.relu2(x)

		return x


class Type2(Layer):
	"""Layer class of NamNet type 2."""

	def __init__(self, filters, **kwargs):
		super(Type2, self).__init__(**kwargs)

		self.type1 = Type1(filters)

	def call(self, inputs):
		x = self.type1(inputs)

		return add([x, inputs])


class Type3(Layer):
	"""Layer class of NamNet type 3."""

	def __init__(self, filters, **kwargs):
		super(Type3, self).__init__(**kwargs)

		self.conv1 = conv2D(filters, (3,3), (1,1))
		self.batchNorm1 = batchNorm()
		self.relu1 = relu()

		self.conv2 = conv2D(filters, (3,3), (1,1))
		self.batchNorm2 = batchNorm()
		self.relu2 = relu()
		self.maxPooling2D = maxPooling2D((3,3), (2,2))

		self.conv3 = conv2D(filters, (1,1), (2,2))
		self.batchNorm3 = batchNorm()

	def call(self, inputs):
		x1 = self.conv1(inputs)
		x1 = self.batchNorm1(x1)
		x1 = self.relu1(x1)

		x1 = self.conv2(x1)
		x1 = self.batchNorm2(x1)
		x1 = self.relu2(x1)
		x1 = self.maxPooling2D(x1)

		x2 = self.conv3(inputs)
		x2 = self.batchNorm3(x2)

		return add([x1, x2])


class Type4(Layer):
	"""Layer class of NamNet type 4."""

	def __init__(self, filters, **kwargs):
		super(Type4, self).__init__(**kwargs)

		self.globalAveragePooling2D = globalAveragePooling2D()
		self.fc = dense(2, use_bias=False, activation='softmax')

	def call(self, inputs):
		x = self.globalAveragePooling2D(inputs)
		x = self.fc(x)

		return x


class NamNet(Model):
	"""NamNet class."""

	def __init__(self, scale=1.0, reg=0.001, **kwargs):
		super(NamNet, self).__init__(**kwargs)
		Networks_functions_keras.SCALE = scale
		Networks_functions_keras.REG = reg

		self.l1_t1 = Type1(16)
		self.l2_t1 = Type1(16)

		self.l3_t2 = Type2(16)
		self.l4_t2 = Type2(16)

		self.l8_t3 = Type3(32)
		self.l9_t3 = Type3(64)
		self.l10_t3 = Type3(128)

		self.l12_t4 = Type4(512)


	def call(self, inputs):
		x = self.l1_t1(inputs)
		x = self.l2_t1(x)

		x = self.l3_t2(x)
		x = self.l4_t2(x)

		x = self.l8_t3(x)
		x = self.l9_t3(x)
		x = self.l10_t3(x)

		x = self.l12_t4(x)

		return x













