import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import Model, Input
from .Networks_functions_keras import *


# Type1 layer
class Type1(Layer):
	def __init__(self, filters, **kwargs):
		super(Type1, self).__init__(**kwargs)

		self.conv = conv2D(filters, (3,3), (1,1))
		self.batchNorm = batchNorm()
		self.relu = ReLU()

	def call(self, inputs):
		x = self.conv(inputs)
		x = self.batchNorm(x)
		x = self.relu(x)

		return x


# Type2 layer
class Type2(Layer):
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


# Type3 layer
class Type3(Layer):
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


# Type4 layer
class Type4(Layer):
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


# SRNet whole layer
class SRNetLayer(Layer):
	def __init__(self, **kwargs):
		super(SRNetLayer, self).__init__(**kwargs)

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

		self.fc = dense(2, use_bias=False, activation='softmax')
		self.softmax = softmax()

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

		x = self.l12_t4(x)

		x = self.fc(x)
		# x = self.softmax(x)

		return x


# SRNet
class SRNet(Model):
	def __init__(self, **kwargs):
		super(SRNet, self).__init__(**kwargs)
		self.network = SRNetLayer()

	def call(self, inputs):
		return self.network(inputs)


# alternative solution
def SRNet_():

	inputs = Input(shape=(256,256,1))
	t1_l1 = Type1(64)(inputs)
	t1_l2 = Type1(16)(t1_l1)

	t2_l3 = Type2(16)(t1_l2)
	t2_l4 = Type2(16)(t2_l3)
	t2_l5 = Type2(16)(t2_l4)
	t2_l6 = Type2(16)(t2_l5)
	t2_l7 = Type2(16)(t2_l6)

	t3_l8 = Type3(16)(t2_l7)
	t3_l9 = Type3(64)(t3_l8)
	t3_l10 = Type3(128)(t3_l9)
	t3_l11 = Type3(256)(t3_l10)

	t4_l12 = Type4(512)(t3_l11)

	fc = dense(2, use_bias=False)(t4_l12)

	outputs = softmax()(fc)

	model = Model(inputs=inputs, outputs=outputs)

	return model












