"""
Fusion module
@authorized by Shasha Bae
@description: define the Fusion
"""

import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import Model, Input

from .Networks_functions import *




class Fusion(Model):
	"""Fusion class."""

	def __init__(self, reg, num_class, **kwargs):
		super(Fusion, self).__init__(**kwargs)
		set_parameters(reg, num_class)

		self.fc = dense(num_class, use_bias=False, activation='softmax')


	def call(self, inputs):

		x = self.fc(inputs)

		return x













