"""
Video retouch detection training module
@authorized by Shasha Bae
@description: train the model for retouch detection
"""

import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import Model, Input
from keras import backend as K
from network.Networks_functions_keras import *

class Alpha_Layer(Layer):
	def __init__(self, **kwargs):
		super(Alpha_Layer, self).__init__(**kwargs)

		w_init = tf.random_normal_initializer()
		self.alpha = self.add_weight(shape=(4,), initializer='random_normal', trainable=True, name='alpha_weights')


	def __call__(self, inputs):
		# print("*************** {}".format(inputs.shape))
		sr = inputs[:,0]
		dct = inputs[:,1]
		# print("*************** {}".format(sr.shape))

		x = tf.math.subtract(sr, dct) # sr - dct
		x = tf.math.multiply(x, self.alpha) # (sr - dct) * alpha
		x = tf.math.add(x, dct) # (sr - dct) * alpha + dct == sr * alpha + (1 - alpha) * dct
		return x


class Alpha_Model(Model):
	def __init__(self, **kwargs):
		super(Alpha_Model, self).__init__(**kwargs)
		self.alpha = Alpha_Layer()
		# self.flatten = Flatten()
		# self.dense1 = Dense(8)
		# self.dense2 = Dense(4)
		# self.softmax = softmax()
		

	def call(self, inputs):
		x = self.alpha(inputs)
		
		# x = self.flatten(inputs)
		# x = self.dense1(x)
		# x = self.dense2(x)
		# x = self.softmax(x)

		return x



def main():



	################################################## Setup the dataset
	labels 			= np.load('./fusion/label_test.npy')
	result_SRNet 	= np.load('./fusion/result_test_SRNet.npy')
	result_DCTNet 	= np.load('./fusion/result_test_DCTNet.npy')
	# index 			= np.load('./fusion/{}_index_test.npy'.format(METHOD))

	assert(labels.shape == result_SRNet.shape == result_DCTNet.shape)

	# split dataset
	labels_train = labels[:int(labels.shape[0] * 0.8)]
	labels_test  = labels[int(labels.shape[0] * 0.8):int(labels.shape[0] * 0.9)]
	labels_valid = labels[int(labels.shape[0] * 0.9):]


	data = np.zeros((result_SRNet.shape[0], 2, 4))
	for idx, (sr, dct) in enumerate(zip(result_SRNet, result_DCTNet)):
		data[idx, 0, :] = sr
		data[idx, 1, :] = dct

	data_train = data[:int(data.shape[0] * 0.8)]
	data_test  = data[int(data.shape[0] * 0.8):int(data.shape[0] * 0.9)]
	data_valid = data[int(data.shape[0] * 0.9):]

	
	model = Alpha_Model()
	model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
	
	model.fit(	data_train, \
				labels_train, \
				epochs=100, \
				batch_size=1024, \
				validation_data=(data_valid, labels_valid))

	model.evaluate(data_test, labels_test, batch_size=256)










if __name__=="__main__":
	# For easy reset of notebook state.
	K.clear_session()  
	main()


