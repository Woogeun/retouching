"""
Fusion module
@authorized by Shasha Bae
@description: define the Fusion
"""

import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import Model, Input

from .Networks_functions import *



"""
SRNet module
@authorized by Shasha Bae
@description: define the SRNet
"""

import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import Model, Input

from .Networks_functions import *




class Type1(Layer):
	"""Layer class of SRNet type 1."""

	def __init__(self, filters, **kwargs):
		super(Type1, self).__init__(**kwargs)

		self.conv = conv2D(filters, (3,3), (1,1))
		self.batchNorm = batchNorm()
		self.relu = relu()


	def call(self, inputs):
		x = self.conv(inputs)
		x = self.batchNorm(x)
		x = self.relu(x)

		return x


class Type2(Layer):
	"""Layer class of SRNet type 2."""

	def __init__(self, filters, **kwargs):
		super(Type2, self).__init__(**kwargs)

		self.type1 = Type1(filters)
		self.conv = conv2D(filters, (3,3), (1,1))
		self.batchNorm = batchNorm()


	def call(self, inputs):
		x = self.type1(inputs)
		x = self.conv(x)
		x = self.batchNorm(x)

		return add([x, inputs])


class Type3(Layer):
	"""Layer class of SRNet type 3."""

	def __init__(self, filters, **kwargs):
		super(Type3, self).__init__(**kwargs)

		self.type1 = Type1(filters)
		self.conv1 = conv2D(filters, (3,3), (1,1))
		self.batchNorm1 = batchNorm()
		self.averagePooling2D = averagePooling2D((3,3), (2,2))

		self.conv2 = conv2D(filters, (3,3), (2,2))
		self.batchNorm2 = batchNorm()


	def call(self, inputs):
		x1 = self.type1(inputs)
		x1 = self.conv1(x1)
		x1 = self.batchNorm1(x1)
		x1 = self.averagePooling2D(x1)

		x2 = self.conv2(inputs)
		x2 = self.batchNorm2(x2)

		return add([x1, x2])


class Type4(Layer):
	"""Layer class of SRNet type 4."""

	def __init__(self, filters, **kwargs):
		super(Type4, self).__init__(**kwargs)

		self.type1 = Type1(filters)
		self.conv = conv2D(filters, (3,3), (1,1))
		self.batchNorm = batchNorm()
		self.globalAveragePooling2D = globalAveragePooling2D()


	def call(self, inputs):
		x = self.type1(inputs)
		x = self.conv(x)
		x = self.batchNorm(x)
		x = self.globalAveragePooling2D(x)

		return x


class SRNet(Layer):
	"""SRNet class."""

	def __init__(self, **kwargs):
		super(SRNet, self).__init__(**kwargs)

		self.l1_t1 = Type1(64)
		self.l2_t1 = Type1(16)

		self.l3_t2 = Type2(16)
		self.l4_t2 = Type2(16)
		self.l5_t2 = Type2(16)
		self.l6_t2 = Type2(16)
		self.l7_t2 = Type2(16)

		self.l8_t3 = Type3(16)
		self.l9_t3 = Type3(64)
		self.l10_t3 = Type3(128)
		self.l11_t3 = Type3(256)

		self.l12_t4 = Type4(512)

		# self.fc = dense(num_class, use_bias=False, activation='softmax')


	def call(self, inputs):
		x = self.l1_t1(inputs)
		x = self.l2_t1(x)

		x = self.l3_t2(x)
		x = self.l4_t2(x)
		x = self.l5_t2(x)
		x = self.l6_t2(x)
		x = self.l7_t2(x)

		x = self.l8_t3(x)
		x = self.l9_t3(x)
		x = self.l10_t3(x)
		x = self.l11_t3(x)

		gap = self.l12_t4(x)
		# x = self.fc(gap)

		return gap




"""
DCTNet module
@authorized by Shasha Bae
@description: define the DCTNet
"""

import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import Model, Input

from .Networks_functions import *




class ConvBlock(Layer):
	"""Layer class of convolutional block of DCTNet."""

	def __init__(self, filters, **kwargs):
		super(ConvBlock, self).__init__(**kwargs)

		self.conv1 = conv2D(filters, (3,3), (1,1))
		self.batchNorm1 = batchNorm()
		self.relu1 = relu()

		self.conv2 = conv2D(filters, (1,1), (1,1))
		self.batchNorm2 = batchNorm()
		self.relu2 = relu()

		self.maxPooling2D = maxPooling2D((2,2), (2,2))

	def call(self, inputs):
		x = self.conv1(inputs)
		x = self.batchNorm1(x)
		x = self.relu1(x)

		x = self.conv2(x)
		x = self.batchNorm2(x)
		x = self.relu2(x)

		x = self.maxPooling2D(x)

		return x


class DCT(Layer):
	"""Layer class of DCT transformation."""

	def __init__(self, **kwargs):
		super(DCT, self).__init__(trainable=False, **kwargs)
		self.dct_basis = load_DCT_basis_64()


	def __call__(self, inputs):
		x = tf.nn.conv2d(inputs, self.dct_basis, strides=[1, 8, 8, 1], padding="SAME")
		gamma = 1e+06

		for b in range(-60,61):
			x3 = tf.divide(tf.reduce_sum(tf.sigmoid(tf.scalar_mul(gamma,tf.subtract(x,b))) ,[1,2]),1024)
			x3 = tf.reshape(x3,[-1,1,64])

			if b==-60:
				x4 = x3
			else:
				x4 = tf.concat([x4,x3],1)

		x5 = x4[:,0:120,:] - x4[:,1:121,:]
		x_dhf = tf.reshape(x5,[-1,120,64,1])

		return x_dhf


class DCTNet(Layer):
	"""DCTNet class."""

	def __init__(self, **kwargs):
		super(DCTNet, self).__init__(**kwargs)

		self.dct = DCT()
		self.dct.build(input_shape=(None,256,256,1))

		self.l1_t1 = ConvBlock(64)
		self.l2_t1 = ConvBlock(128)
		self.l3_t1 = ConvBlock(256)
		self.l4_t1 = ConvBlock(512)

		self.globalAveragePooling2D = globalAveragePooling2D()

		# self.fc3 = dense(num_class, use_bias=False, activation='softmax')


	def call(self, inputs):
		x = self.dct(inputs)

		x = self.l1_t1(x)
		x = self.l2_t1(x)
		x = self.l3_t1(x)
		x = self.l4_t1(x)

		gap = self.globalAveragePooling2D(x)

		# x = self.fc3(gap)

		return gap
























class Total(Model):
	"""Fusion class."""

	def __init__(self, reg, num_class, **kwargs):
		super(Total, self).__init__(**kwargs)
		set_parameters(reg, num_class)

		self.srnet = SRNet()
		self.dctnet = DCTNet()

		self.fc = dense(num_class, use_bias=False, activation='softmax')


	def call(self, inputs):

		x1 = self.srnet(inputs)
		x2 = self.dctnet(inputs)
		x = tf.concat([x1, x2], axis=-1)

		x = self.fc(x)

		return x













