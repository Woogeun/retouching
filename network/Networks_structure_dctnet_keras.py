"""
DCTNet module
@authorized by Shasha Bae
@description: define the DCTNet
"""

import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import Model, Input

from . import Networks_functions_keras
from .Networks_functions_keras import *



class Type1(Layer):
	"""Layer class of DCTNet type 1."""

	def __init__(self, filters, **kwargs):
		super(Type1, self).__init__(**kwargs)

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
		# print(self.dct_basis.shape)

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


class DCTNet(Model):
	"""DCTNet class."""

	def __init__(self, scale=1.0, reg=0.001, num_class=2, **kwargs):
		super(DCTNet, self).__init__(**kwargs)
		Networks_functions_keras.SCALE = scale
		Networks_functions_keras.REG = reg

		# 
		self.dct = DCT()
		self.dct.build(input_shape=(None,256,256,1))
		self.l1_t1 = Type1(64)
		self.l2_t1 = Type1(128)
		self.l3_t1 = Type1(256)
		self.l4_t1 = Type1(512)
		self.globalAveragePooling2D = globalAveragePooling2D()
		# self.flt = flatten()
		# self.fc1 = dense(512, use_bias=True, activation='relu')
		# self.fc2 = dense(256, use_bias=True, activation='relu')
		self.fc3 = dense(num_class, use_bias=False, activation='softmax')

	def call(self, inputs):
		x = self.dct(inputs)
		x = self.l1_t1(x)
		x = self.l2_t1(x)
		x = self.l3_t1(x)
		x = self.l4_t1(x)
		x = self.globalAveragePooling2D(x)
		# x = self.flt(x)
		# x = self.fc1(x)
		# x = self.fc2(x)
		x = self.fc3(x)
		return x













