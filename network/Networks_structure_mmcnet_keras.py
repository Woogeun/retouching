"""
MMCNet module
@authorized by Shasha Bae
@description: define the MMCNet
"""

import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import Model, Input

from . import Networks_functions_keras
from .Networks_functions_keras import *




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


class DHF(Layer):
	"""Layer class of DHF extractor."""

	def __init__(self, **kwargs):
		super(DHF, self).__init__(trainable=False, **kwargs)
		self.dct_basis = load_DCT_basis_64()

	def __call__(self, inputs):
		dct = tf.nn.conv2d(inputs, self.dct_basis, strides=[1, 8, 8, 1], padding="SAME") # dct.shape == (None, 32, 32, 64)
		gamma = 1e+06
		for b in range(-60, 61):
			patch = tf.divide(tf.reduce_sum(tf.sigmoid(tf.scalar_mul(gamma, tf.subtract(dct, b))), [1, 2]), 1024)
			patch = tf.reshape(patch,[-1, 1, 64])
			if b==-60:
				cummulative_dct_histogram = patch
			else:
				cummulative_dct_histogram = tf.concat([cummulative_dct_histogram,patch], 1)

		dct_histogram = cummulative_dct_histogram[:, 0:120, :] - cummulative_dct_histogram[:, 1:121, :]

		dhf = tf.reshape(dct_histogram,[-1, 120, 64, 1])

		return dhf


class Type1_DHF(Layer):
	"""Layer class of DHF type 1."""

	def __init__(self, filters, **kwargs):
		super(Type1_DHF, self).__init__(**kwargs)

		self.conv1 = conv2D_(filters, (3,3), (1,1))
		self.batchNorm1 = batchNorm()
		self.relu1 = relu()

		self.conv2 = conv2D_(filters, (1,1), (1,1))
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



# SRNet : DHFNet = 1 : 1
class MMCNet(Model):
	"""MMCNet class."""

	def __init__(self, scale=1.0, reg=0.001, num_class=2, **kwargs):
		super(MMCNet, self).__init__(**kwargs)
		Networks_functions_keras.SCALE = scale
		Networks_functions_keras.REG = reg

		# SRNet path
		self.srnet_l1_t1 = Type1(64)
		self.srnet_l2_t1 = Type1(16)

		self.srnet_l3_t2 = Type2(16)
		self.srnet_l4_t2 = Type2(16)
		self.srnet_l5_t2 = Type2(16)
		self.srnet_l6_t2 = Type2(16)
		self.srnet_l7_t2 = Type2(16)

		self.srnet_l8_t3 = Type3(16)
		self.srnet_l9_t3 = Type3(32)
		self.srnet_l10_t3 = Type3(64)
		self.srnet_l11_t3 = Type3(128)

		self.srnet_l12_t4 = Type4(512)


		# DCT Histogram Feature path
		self.dhf = DHF()
		self.dhf_l1_t1 = Type1_DHF(16)
		self.dhf_l2_t1 = Type1_DHF(32)
		self.dhf_l3_t1 = Type1_DHF(64)

		self.dhf_flt = flatten()
		self.dhf_fc = dense(64)


		# Concatenater
		self.concatenate = concatenate()
		self.fc1 = dense(256, use_bias=True, activation='relu')
		self.fc2 = dense(128, use_bias=True, activation='relu')
		self.fc3 = dense(num_class, use_bias=False, activation='softmax')


	def call(self, inputs):
		# SRNet path
		srnet_path = self.srnet_l1_t1(inputs)
		srnet_path = self.srnet_l2_t1(srnet_path)

		srnet_path = self.srnet_l3_t2(srnet_path)
		srnet_path = self.srnet_l4_t2(srnet_path)
		srnet_path = self.srnet_l5_t2(srnet_path)
		srnet_path = self.srnet_l6_t2(srnet_path)
		srnet_path = self.srnet_l7_t2(srnet_path)

		srnet_path = self.srnet_l8_t3(srnet_path)
		srnet_path = self.srnet_l9_t3(srnet_path)
		srnet_path = self.srnet_l10_t3(srnet_path)
		srnet_path = self.srnet_l11_t3(srnet_path)

		srnet_path = self.srnet_l12_t4(srnet_path) # (?, 256)


		# DCT Histogram Feature path
		dhf_path = self.dhf(inputs)
		dhf_path = self.dhf_l1_t1(dhf_path)
		dhf_path = self.dhf_l2_t1(dhf_path)
		dhf_path = self.dhf_l3_t1(dhf_path)

		dhf_path = self.dhf_flt(dhf_path) # (?, 30720)
		# dhf_path = self.dhf_fc(dhf_path) # (?, 512)

		


		
		
		
		# Concatenater
		x = self.concatenate([srnet_path, dhf_path])
		x = self.fc1(x)
		x = self.fc2(x)
		x = self.fc3(x)

		return x

'''
# SRNet Noise + MISLNet Feature + MISLNet Classifier
class MMCNet(Model):
	"""MMCNet class."""

	def __init__(self, scale=1.0, reg=0.001, num_class=2, **kwargs):
		super(MMCNet, self).__init__(**kwargs)
		Networks_functions_keras.SCALE = scale
		Networks_functions_keras.REG = reg

		# SRNet path
		self.srnet_l1_t1 = Type1(64)
		self.srnet_l2_t1 = Type1(16)

		self.srnet_l3_t2 = Type2(16)
		self.srnet_l4_t2 = Type2(16)
		self.srnet_l5_t2 = Type2(16)
		self.srnet_l6_t2 = Type2(16)
		self.srnet_l7_t2 = Type2(16)


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
		# SRNet path
		x = self.srnet_l1_t1(inputs)
		x = self.srnet_l2_t1(x)

		x = self.srnet_l3_t2(x)
		x = self.srnet_l4_t2(x)
		x = self.srnet_l5_t2(x)
		x = self.srnet_l6_t2(x)
		x = self.srnet_l7_t2(x)

		x = self.block1(x)
		x = self.block2(x)
		x = self.block3(x)
		x = self.block4(x)

		x = self.flatten(x)
		x = self.fc1(x)
		x = self.fc2(x)
		x = self.fc3(x)

		return x


# MISLNet Noise + SRNet Feature + MISLNet Classifier
class MMCNet(Model):
	"""MMCNet class."""

	def __init__(self, scale=1.0, reg=0.001, num_class=2, **kwargs):
		super(MMCNet, self).__init__(**kwargs)
		Networks_functions_keras.SCALE = scale
		Networks_functions_keras.REG = reg

		# MISLNet Noise
		self.conv 	= ConstrainedLayer(3, (5,5), (1,1))


		# SRNet Noise Feature
		self.srnet_l8_t3 = Type3(16)
		self.srnet_l9_t3 = Type3(64)
		self.srnet_l10_t3 = Type3(128)
		self.srnet_l11_t3 = Type3(256)

		self.srnet_l12_t4 = Type4(512)


		# classification
		self.flatten 	= flatten()
		self.fc1 		= dense(200, use_bias=False, activation='tanh')
		self.fc2 		= dense(200, use_bias=False, activation='tanh')
		self.fc3 		= dense(num_class, use_bias=False, activation='softmax')



	def call(self, inputs):
		x = self.conv(inputs)

		x = self.srnet_l8_t3(x)
		x = self.srnet_l9_t3(x)
		x = self.srnet_l10_t3(x)
		x = self.srnet_l11_t3(x)

		x = self.srnet_l12_t4(x) 

		x = self.flatten(x)
		x = self.fc1(x)
		x = self.fc2(x)
		x = self.fc3(x)

		return x
'''










