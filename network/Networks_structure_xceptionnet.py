"""
XceptionNet module
@authorized by Shasha Bae
@description: define the XceptionNet
"""

import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import Model, Input

from .Networks_functions import *




class EntryFlow(Layer):
	"""Entry Layer of XceptionNet type 1."""

	def __init__(self, filters, use_relu=True, **kwargs):
		super(EntryFlow, self).__init__(**kwargs)
		self.use_relu = use_relu

		self.relu1 = relu()
		self.separableConv2D1 = separableConv2D(filters, (3,3), (1,1))
		self.batchNorm1 = batchNorm()
		
		self.relu2 = relu()
		self.separableConv2D2 = separableConv2D(filters, (3,3), (1,1))
		self.batchNorm2 = batchNorm()
		self.maxPooling2D2 = maxPooling2D((3,3), (2,2))

		self.conv3 = conv2D(filters, (1,1), (2,2))
		self.batchNorm3 = batchNorm()


	def call(self, inputs):
		if self.use_relu:
			x1 = self.relu1(inputs)
			x1 = self.separableConv2D1(x1)
		else:
			x1 = self.separableConv2D1(inputs)

		x1 = self.batchNorm1(x1)
		
		x1 = self.relu2(x1)
		x1 = self.separableConv2D2(x1)
		x1 = self.batchNorm2(x1)
		x1 = self.maxPooling2D2(x1)

		x2 = self.conv3(inputs)
		x2 = self.batchNorm3(x2)

		return add([x1, x2])


class MiddleFlow(Layer):
	"""Middle layer of XceptionNet type 2."""

	def __init__(self, filters, **kwargs):
		super(MiddleFlow, self).__init__(**kwargs)

		self.relu1 = relu()
		self.separableConv2D1 = separableConv2D(filters, (3,3), (1,1))
		self.batchNorm1 = batchNorm()

		self.relu2 = relu()
		self.separableConv2D2 = separableConv2D(filters, (3,3), (1,1))
		self.batchNorm2 = batchNorm()

		self.relu3 = relu()
		self.separableConv2D3 = separableConv2D(filters, (3,3), (1,1))
		self.batchNorm3 = batchNorm()


	def call(self, inputs):
		x = self.relu1(inputs)
		x = self.separableConv2D1(x)
		x = self.batchNorm1(x)

		x = self.relu2(x)
		x = self.separableConv2D2(x)
		x = self.batchNorm2(x)

		x = self.relu3(x)
		x = self.separableConv2D3(x)
		x = self.batchNorm3(x)

		return add([x, inputs])


class ExitFlow(Layer):
	"""Layer class of XceptionNet type 3."""

	def __init__(self, filters, **kwargs):
		super(ExitFlow, self).__init__(**kwargs)

		self.relu1 = relu()
		self.separableConv2D1 = separableConv2D(512, (3,3), (1,1))
		self.batchNorm1 = batchNorm()

		self.relu2 = relu()
		self.separableConv2D2 = separableConv2D(filters, (3,3), (1,1))
		self.batchNorm2 = batchNorm()
		self.maxPooling2D2 = maxPooling2D((3,3), (2,2))

		self.conv3 = conv2D(filters, (1,1), (2,2))
		self.batchNorm3 = batchNorm()

		self.separableConv2D4 = separableConv2D(1024, (3,3), (1,1))
		self.batchNorm4 = batchNorm()
		self.relu4 = relu()

		self.separableConv2D5 = separableConv2D(2048, (3,3), (1,1))
		self.batchNorm5 = batchNorm()
		self.relu5 = relu()

		self.globalAveragePooling2D = globalAveragePooling2D()



	def call(self, inputs):
		x1 = self.relu1(inputs)
		x1 = self.separableConv2D1(x1)
		x1 = self.batchNorm1(x1)

		x1 = self.relu2(x1)
		x1 = self.separableConv2D2(x1)
		x1 = self.batchNorm2(x1)
		x1 = self.maxPooling2D2(x1)

		x2 = self.conv3(inputs)
		x2 = self.batchNorm3(x2)

		x = add([x1, x2])

		x = self.separableConv2D4(x)
		x = self.batchNorm4(x)
		x = self.relu4(x)

		x = self.separableConv2D5(x)
		x = self.batchNorm5(x)
		x = self.relu5(x)

		x = self.globalAveragePooling2D(x)

		return x


class XceptionNet(Model):
	"""XceptionNet class."""

	def __init__(self, reg, num_class, **kwargs):
		super(XceptionNet, self).__init__(**kwargs)
		set_parameters(reg, num_class)

		self.conv0 = Conv2D(1, (1,1), (2,2), padding='same')
		self.batchNorm0 = batchNorm()
		self.relu0 = relu()

		self.conv1 = Conv2D(32, (3,3), (2,2), padding='valid')
		self.batchNorm1 = batchNorm()
		self.relu1 = relu()

		self.conv2 = Conv2D(64, (3,3), (1,1), padding='valid')
		self.batchNorm2 = batchNorm()
		self.relu2 = relu()

		self.entry1 = EntryFlow(128, use_relu=False)
		self.entry2 = EntryFlow(256)
		# self.entry3 = EntryFlow(256)

		self.middle1 = MiddleFlow(256)
		self.middle2 = MiddleFlow(256)
		self.middle3 = MiddleFlow(256)
		self.middle4 = MiddleFlow(256)
		self.middle5 = MiddleFlow(256)
		self.middle6 = MiddleFlow(256)
		self.middle7 = MiddleFlow(256)
		self.middle8 = MiddleFlow(256)

		self.exit = ExitFlow(768)

		self.fc = dense(num_class, use_bias=False, activation='softmax')


	def call(self, inputs):

		x = self.conv0(inputs)
		x = self.batchNorm0(x)
		x = self.relu0(x)

		x = self.conv1(x)
		x = self.batchNorm1(x)
		x = self.relu1(x)

		x = self.conv2(x)
		x = self.batchNorm2(x)
		x = self.relu2(x)

		x = self.entry1(x)
		x = self.entry2(x)
		# x = self.entry3(x)

		x = self.middle1(x)
		x = self.middle2(x)
		x = self.middle3(x)
		x = self.middle4(x)
		x = self.middle5(x)
		x = self.middle6(x)
		x = self.middle7(x)
		x = self.middle8(x)

		x = self.exit(x)

		x = self.fc(x)

		return x













