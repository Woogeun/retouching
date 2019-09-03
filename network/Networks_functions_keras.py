"""
Tensorflow keras functions
@authorized by Shasha Bae
@description: tensorflow keras helper functions for logging setup, dataset configuration, training, and network layers 
"""

import random
from datetime import datetime
from os import makedirs, cpu_count
from os.path import join

import numpy as np

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers, activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras.callbacks import Callback, ModelCheckpoint, TensorBoard, LearningRateScheduler, EarlyStopping




##################### Network parameters
# This is the default internal TF session used by Keras.
# It can be set manually via `set_session(sess)`.
SCALE = 1.0
REG = 0.001


##################### Simple helper functions
def print_args(args):
	"""Print logs for arguments

   	# Arguments
    	args: the argparser parsed object
	"""

	args_dict = vars(args)
	for key, value in args_dict.items():
		print("{:20s}: {}\n".format(key, value))


def write_args(args, filepath):
	"""Write logs for arguments to the file

   	# Arguments
    	args: the argparser parsed object
    	filepath: the log file path to be written
	"""

	args_dict = vars(args)
	with open(filepath, 'w') as f:
		for key, value in args_dict.items():
			f.write("{:20s}: {}\n".format(key, value))


def write_history(history, filepath):
	"""Write training history to the file

   	# Arguments
    	history: the history object returned from tf.keras.model.fit()
    	filepath: the history file path to be written
	"""

	with open(filepath, 'w') as f:
		for key, values in history.history.items():
			f.write("{}\n".format(key))
			for value in values:
				f.write("{:0.5f}\n".format(value))
			f.write("\n")


def write_result(keys, values, filepath):
	"""Write test result to the file

   	# Arguments
   		keys: the string list of metrics of the model
    	values: the result value returned from tf.keras.model.evaluate()
    	filepath: the result file path to be written
	"""

	with open(filepath, 'w') as f:
		for key, value in zip(keys, values):
			f.write("{:20s}: {:0.5f}\n".format(key, value))



##################### Dataset parsing functions
def _bytes_to_array(features, key, element_type, dimension):
	"""Convert bytes data into tensorflow array

   	# arguments
   		features: tfrecord data
   		key: string label
    	element_type: the tensorflow data type for cast
    	dimension: list or numpy array or tensor for conversion dimension

	# Returns
		A tensorflow data structure converted from bytes data
	"""

	return 	tf.cast(\
				tf.reshape(\
					tf.decode_raw(\
						features[key],\
						element_type),\
					dimension) ,\
				tf.float32)


def _parse_function(example_proto):
	"""Parse the example prototype data to tensorflow data structure

   	# arguments
   		example_proto: tensorflow example prototype object

	# Returns
		A pair of frames data and string label
	"""

	feature_description = {
		# mendatory informations
		'frames': tf.FixedLenFeature([], dtype=tf.string, default_value=""),
		'label'	: tf.FixedLenFeature([], dtype=tf.string, default_value=""),

		# additional information
		# 'br'	: tf.FixedLenFeature([], dtype=tf.int64, default_value=1)
	}

	# parse feature
	features = tf.parse_single_example(example_proto, feature_description)

	frames = _bytes_to_array(features, 'frames', tf.uint8, [256, 256, 3])
	frames = tf.image.rgb_to_grayscale(frames)
	label = _bytes_to_array(features, 'label', tf.uint8, [2])

	# return frames, label, br
	return frames, label


def configure_dataset(fnames, batch_size):
	"""Configure the dataset 

   	# arguments
   		fnames: the list of strings of tfrecord file names
   		batch_size: the int represents batch size

	# Returns
		A dataset object configured by batch size
	"""

	random.shuffle(fnames)
	buffer_size = max(len(fnames) / batch_size, 16) # recommend buffer_size = # of elements / batches
	buffer_size = tf.cast(buffer_size, tf.int64)

	dataset = tf.data.TFRecordDataset(fnames)
	dataset = dataset.map(_parse_function, num_parallel_calls=cpu_count())
	dataset = dataset.prefetch(buffer_size=buffer_size) 
	dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True) 
	dataset = dataset.repeat()
	dataset = dataset.batch(batch_size)

	return dataset



##################### Manage callback classes
# custom model checkpoint save callback
class SaveWeight(Callback):
	"""Callback class for Saving the trained weights."""

	def __init__(self, ckpt_path, **kwargs):
		super(SaveWeight, self).__init__(**kwargs)
		self.ckpt_path = ckpt_path

	def on_epoch_end(self, epoch, logs):
		self.model.save_weights(join(self.ckpt_path, "weights_{}".format(epoch)), save_format='h5')


# custom tensorboard callback
class TrainValTensorBoard(TensorBoard):
	"""Callback class for logging the train and validation data in one graph."""

	def __init__(self, log_dir='./logs', **kwargs):
		self.val_log_dir = join(log_dir, 'validation')
		training_log_dir = join(log_dir, 'training')
		super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

	def set_model(self, model):
		if context.executing_eagerly():
			self.val_writer = tf.contrib.summary.create_file_writer(self.val_log_dir)
		else:
			self.val_writer = tf.summary.FileWriter(self.val_log_dir)
		super(TrainValTensorBoard, self).set_model(model)

	def _write_custom_summaries(self, step, logs=None):
		logs = logs or {}
		val_logs = {k.replace('val_', ''): v for k, v in logs.items() if 'val_' in k}
		if context.executing_eagerly():
			with self.val_writer.as_default(), tf.contrib.summary.always_record_summaries():
				for name, value in val_logs.items():
					tf.contrib.summary.scalar(name, value.item(), step=step)
		else:
			for name, value in val_logs.items():
				summary = tf.Summary()
				summary_value = summary.value.add()
				summary_value.simple_value = value.item()
				summary_value.tag = name
				self.val_writer.add_summary(summary, step)
		self.val_writer.flush()

		logs = {k: v for k, v in logs.items() if not 'val_' in k}
		super(TrainValTensorBoard, self)._write_custom_summaries(step, logs)

	def on_train_end(self, logs=None):
		super(TrainValTensorBoard, self).on_train_end(logs)
		self.val_writer.close()


# custom learning rate scheduler callback
class CustomLearningRateScheduler(LearningRateScheduler):
	"""Callback class for updating the learning rate per batch."""

	def __init__(self, schedule, verbose, LR_UPDATE_INTERVAL, LR_UPDATE_RATE, mode="iter"):
		self.LR_UPDATE_INTERVAL = LR_UPDATE_INTERVAL
		self.LR_UPDATE_RATE = LR_UPDATE_RATE
		self.mode = mode
		if self.mode == "epoch":
			self.LR_UPDATE_INTERVAL = 1
		self.iteration = 0

		super(CustomLearningRateScheduler, self).__init__(schedule, verbose)


	def on_batch_begin(self, batch, logs=None):
		if self.mode == "iter":
			if not hasattr(self.model.optimizer, 'lr'):
				raise ValueError('Optimizer must have a "lr" attribute.')

			self.iteration += 1
			lr = float(K.get_value(self.model.optimizer.lr))
			lr = self.schedule(self.iteration, lr, self.LR_UPDATE_INTERVAL, self.LR_UPDATE_RATE)
			
			if not isinstance(lr, (float, np.float32, np.float64)):
				raise ValueError('The output of the "schedule" function should be float.')

			K.set_value(self.model.optimizer.lr, lr)


	def on_batch_end(self, batch, logs=None):
		if self.mode == "iter":
			logs = logs or {}
			logs['lr'] = K.get_value(self.model.optimizer.lr)


	def on_epoch_begin(self, epoch, logs=None):
		if self.mode == "epoch":
			if not hasattr(self.model.optimizer, 'lr'):
				raise ValueError('Optimizer must have a "lr" attribute.')

			lr = float(K.get_value(self.model.optimizer.lr))
			lr = self.schedule(epoch, lr, self.LR_UPDATE_INTERVAL, self.LR_UPDATE_RATE)
			
			if not isinstance(lr, (float, np.float32, np.float64)):
				raise ValueError('The output of the "schedule" function should be float.')

			K.set_value(self.model.optimizer.lr, lr)

		# Log learning rate information
		lr = float(K.get_value(self.model.optimizer.lr))
		if self.verbose > 0:
			print('\n%05d: LearningRateScheduler reducing learning rate to %10f.' % (((epoch + 1) if self.mode == "epoch" else (self.iteration + 1)), lr))


	def on_epoch_end(self, epoch, logs=None):
		if self.mode == "epoch":
			logs = logs or {}
			logs['lr'] = K.get_value(self.model.optimizer.lr)

def lr_scheduler(iteration, lr, LR_UPDATE_INTERVAL, LR_UPDATE_RATE):
	"""Determine the next learning rate 

   	# arguments
   		iteration: the current number of batch already trained
   		lr: current learning rate
   		LR_UPDATE_INTERVAL: the batch_wise update interval 
   		LR_UPDATE_RATE: the value which to be multiplied to the current learning rate

	# Returns
		The next learning rate
	"""
	if iteration % LR_UPDATE_INTERVAL == 0:
		lr *= LR_UPDATE_RATE 
	return lr



# get weight callback
class GetWeight(Callback):
	"""Callback class for debugging the first layer weights by print."""

	def __init__(self, **kwargs):
		super(GetWeight, self).__init__(**kwargs)

	def on_epoch_begin(self, epoch, logs):
		print(self.model.layers[0].get_weights()[0])



def load_callbacks(args):
	"""Return the callback list

   	# arguments
   		args: the object of parsed argparse

	# Returns
		A list of tf.keras.callbacks 
	"""

	LOG_PATH 			= args.log_path
	METHOD 				= args.method
	BATCH_SIZE 			= args.batch_size
	LR_UPDATE_INTERVAL 	= args.lr_update_interval
	LR_UPDATE_RATE 		= args.lr_update_rate

	# Set log file path
	current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
	LOG_PATH = join(LOG_PATH, current_time + "_{}".format("all" if METHOD=="*" else METHOD))


	# 1. checkpoint callback
	ckpt_path = join(LOG_PATH, "checkpoint")
	makedirs(ckpt_path)
	ckpt_callback = SaveWeight(ckpt_path)
	

	# 2. tensorboard callback
	tb_callback = TrainValTensorBoard(log_dir=LOG_PATH, batch_size=BATCH_SIZE, update_freq='batch')


	# 3. learning rate scheduler callback
	lr_callback = CustomLearningRateScheduler(	schedule=lr_scheduler, \
												verbose=1, \
												LR_UPDATE_INTERVAL=LR_UPDATE_INTERVAL, \
										 		LR_UPDATE_RATE=LR_UPDATE_RATE, \
										 		mode="epoch")


	# 4. early stop callback
	earlyStop_callback = EarlyStopping(	monitor='val_loss', \
										min_delta=1e-2, \
										patience=3, \
										verbose=1)


	# 5. print first layer callback for debug
	getWeight_callback = GetWeight()
	

	return LOG_PATH, [	ckpt_callback, \
						tb_callback, \
						lr_callback, \
						# earlyStop_callback, \
						# getWeight_callback, \
						]



##################### Custom constraint objects
# Custum constraint
class CustomConstraint(constraints.Constraint):
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

	return layers.Conv2D(filters=filters, \
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
	kernel_initializer 	= tf.keras.initializers.random_normal(mean=0, stddev=0.01)
	bias_initializer 	= 'zeros'
	kernel_regularizer 	= None
	bias_regularizer 	= None

	return layers.Dense(units=units, \
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

	return layers.MaxPool2D(pool_size=pool_size, \
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

	return layers.AveragePooling2D(	pool_size=pool_size, \
									strides=strides, \
									padding=padding, \
									data_format=data_format)

def globalAveragePooling2D():
	"""Global average pooling layer

	# Returns
		A tf.keras.layers.GlobalAveragePooling2D 
	"""

	return layers.GlobalAveragePooling2D()


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

	return layers.BatchNormalization(axis=-axis, \
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

	return layers.Flatten()

def add(*args):
	"""Add the arguments layers

   	# arguments
   		*args: the list of tf.keras.layers

	# Returns
		The graph of added layer
	"""

	return layers.Add()(*args)






