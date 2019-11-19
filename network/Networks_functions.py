"""
Tensorflow keras functions
@authorized by Shasha Bae
@description: tensorflow keras helper functions for logging setup, dataset configuration, training, and network layers 
"""

from math import sqrt, pi, cos

import numpy as np

import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import activations
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.constraints import Constraint




##################### Network parameters
# These are the default train parameters of network models.
# It can be set manually via `Networks_functions_keras.SCALE = 0.5`.
SCALE = None
REG = None
NUM_CLASS = None

def set_parameters(scale, reg, num_class):
	"""Set gloval variables with given parameters

   	# arguments
   		scale: the scale of convolutional channel (0.0 < scale <= 1.0)
   		reg: the regularization term (0.0 <= reg <= 1.0)
   		num_class: the number of class for classification

	"""
	global SCALE, REG, NUM_CLASS
	SCALE, REG, NUM_CLASS = scale, reg, num_class



##################### Custom constraint objects
# Custum constraint
class CustomConstraint(Constraint):
	"""Constraint class for first layer of Bayar network."""

	def __init__(self, sum=1, center=-1, axis=None):
		self.sum = sum
		self.center = center
		self.axis = axis

	def __call__(self, w):
		# get center coordinates
		shape = w.get_shape().as_list()
		assert(shape[0] % 2 == 1)
		assert(shape[1] % 2 == 1)
		centerX_ = (shape[0]-1)/2
		centerY_ = (shape[1]-1)/2
		centerX = tf.cast(centerX_, dtype=tf.int64)
		centerY = tf.cast(centerY_, dtype=tf.int64)

		# get impulse tensor which has same center value with w and other values are 0
		centerValue = w[centerX, centerY]
		impulse = K.zeros(shape)
		impulse = impulse[centerX, centerY].assign(centerValue)

		# get impulse tensor which has center value is -1 and other values are 0
		minus_ones = self.center * tf.constant(np.ones(shape), dtype=tf.float32)
		impulse_ = K.zeros(shape)
		impulse_ = impulse_[centerX, centerY].assign(minus_ones[centerX, centerY])


		# set center value to zero
		w -= impulse


		# normalize
		w /= K.sum(w, axis=self.axis) / self.sum


		# set center value to -1
		w += impulse_

		return w



	def get_config(self):

		return {'sum': self.sum, 'center': self.center, 'axis': self.axis}




##################### Network layer functions
# weights
def conv2D(filters, kernel_size, strides=(1,1), kernel_constraint=None):
	"""2D convolution layer

   	# arguments
   		filters: the number of filters
   		kernel_size: a tuple of kernel size
   		strides: a tuple of strides
   		kernel_constraint: the tf.keras.constraints object 

	# Returns
		A tf.keras.layers.Conv2D layer
	"""

	filters 			= int(SCALE * filters)
	padding 			= 'same'
	data_format 		= 'channels_last'
	activation 			= None
	use_bias 			= True
	kernel_initializer 	= tf.keras.initializers.he_normal()
	bias_initializer 	= tf.keras.initializers.constant(value=0.2)
	kernel_regularizer 	= tf.keras.regularizers.l2(l=REG)
	bias_regularizer 	= None

	return Conv2D(	filters=filters, \
					kernel_size=kernel_size, \
					strides=strides, \
					padding=padding, \
					data_format=data_format, \
					activation=activation, \
					use_bias=use_bias, \
					kernel_initializer=kernel_initializer, \
					bias_initializer=bias_initializer, \
					kernel_regularizer=kernel_regularizer, \
					bias_regularizer=bias_regularizer,
					kernel_constraint=kernel_constraint)

def dense(units, use_bias=True, activation=None):
	"""Fully connected layer

   	# arguments
   		units: the number of output units
   		use_bias: bool whether use bias term or not
   		activation: the activation function after fully connected layer

	# Returns
		A tf.keras.layers.Dense layer
	"""

	activation 			= activation
	use_bias			= use_bias
	kernel_initializer 	= tf.keras.initializers.he_normal()
	bias_initializer 	= 'zeros'
	kernel_regularizer 	= None
	bias_regularizer 	= None

	return Dense(	units=units, \
					activation=activation, \
					use_bias=use_bias, \
					kernel_initializer=kernel_initializer, \
					bias_initializer=bias_initializer, \
					kernel_regularizer=kernel_regularizer, \
					bias_regularizer=bias_regularizer)


# activations
def relu():
	"""ReLU activation function

	# Returns
		A tf.keras.activations.relu
	"""

	return activations.relu

def softmax():
	"""Softmax activation function

	# Returns
		A tf.keras.activations.softmax
	"""

	return activations.softmax

def tanh():
	"""Tanh activation function

	# Returns
		A tf.keras.activations.tanh
	"""

	return activations.tanh

def sigmoid():
	"""Sigmoid activation function

	# Returns
		A tf.keras.activations.sigmoid
	"""

	return activations.sigmoid


# pooling
def maxPooling2D(pool_size, strides=(1,1)):
	"""Max pooling layer

   	# arguments
   		pool_size: a tuple of pooling window
   		strides: a tuple of strides

	# Returns
		A tf.keras.layers.MaxPooling2D 
	"""

	padding 	= 'same'

	return MaxPool2D(	pool_size=pool_size, \
						strides=strides, \
						padding=padding)

def averagePooling2D(pool_size, strides=(1,1)):
	"""Average pooling layer

   	# arguments
   		pool_size: a tuple of pooling window
   		strides: a tuple of strides

	# Returns
		A tf.keras.layers.AveragePooling2D 
	"""

	padding 	= 'same'
	data_format = None

	return AveragePooling2D(pool_size=pool_size, \
							strides=strides, \
							padding=padding, \
							data_format=data_format)

def globalAveragePooling2D():
	"""Global average pooling layer

	# Returns
		A tf.keras.layers.GlobalAveragePooling2D 
	"""

	return GlobalAveragePooling2D()


# manipulation
def batchNorm():
	"""Batch normalization layer

	# Returns
		A tf.keras.layers.BatchNormalization
	"""

	axis 						= -1
	momentum 					= 0.9
	epsilon 					= 0.001
	center 						= True
	scale 						= True
	beta_initializer 			= 'zeros'
	gamma_initializer 			= 'ones'
	moving_mean_initializer 	= 'zeros'
	moving_variance_initializer = 'ones'
	beta_regularizer 			= None
	gamma_regularizer 			= None
	beta_constraint 			= None
	gamma_constraint 			= None
	trainable 					= True
	virtual_batch_size 			= None

	return BatchNormalization(	axis=-axis, \
								momentum=momentum, \
								epsilon=epsilon, \
								center=center, \
								scale=scale, \
								beta_initializer=beta_initializer, \
								gamma_initializer=gamma_initializer, \
								moving_mean_initializer=moving_mean_initializer, \
								moving_variance_initializer=moving_variance_initializer, \
								beta_regularizer=beta_regularizer, \
								gamma_regularizer=gamma_regularizer, \
								beta_constraint=beta_constraint, \
								gamma_constraint=gamma_constraint, \
								trainable=trainable, \
								virtual_batch_size=virtual_batch_size)

def flatten():
	"""Flatten the 2D convolution layer into 1D units layer

	# Returns
		A tf.keras.layers.Faltten
	"""

	return Flatten()

def concatenate():
	"""Concatenate input tensors

	# Returns
		A tf.keras.layers.Concatenate
	"""

	return Concatenate()

def add(*args):
	"""Add the arguments layers

   	# arguments
   		*args: the list of tf.keras.layers

	# Returns
		The graph of added layer
	"""

	return Add()(*args)

def dropout(rate):
	"""drouput layer

	# Returns
		A tf.keras.layers.Faltten
	"""

	return Dropout(rate)


#caclulate DCT basis
def cal_scale(p,q): 
	""" 8x8 dct scale calculation for cal_basis function

	# arguments
   		p: horizontal index 
   		q: vertical index

	# Returns
		horizontal scale, vertical scale
	"""

	if p==0:
		ap = 1/(sqrt(8))
	else:
		ap = sqrt(0.25) #0.25 = 2/8
	if q==0:
		aq = 1/(sqrt(8))
	else:
		aq = sqrt(0.25) #0.25 = 2/8

	return ap,aq

def cal_basis(p,q):
	""" 8x8 dct basis calculation for load_DCT_basis_64 function

	# arguments
   		p: horizontal index 
   		q: vertical index

	# Returns
		8x8 basis for given horizontal and vertical index
	"""

	basis = np.zeros((8,8))
	ap,aq = cal_scale(p,q)
	for m in range(0,8):
		for n in range(0,8):
			basis[m,n] = ap*aq*cos(pi*(2*m+1)*p/16)*cos(pi*(2*n+1)*q/16)

	return basis

def load_DCT_basis_64():
	"""drouput layer

	# Returns
		A 8x8x1x64 tf.constant initialized by 8x8 DCT basis
	"""

	basis_64 = np.zeros((8,8,1,64))
	idx = 0
	for i in range(8):
		for j in range(8):
			basis_64[:,:,0,idx] = cal_basis(i,j)
			idx = idx + 1

	return tf.constant(basis_64.tolist())


