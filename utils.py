"""
Util functions
@authorized by Shasha Bae
@description: util functions for printing writing training environments
"""

import sys
import random
from os import makedirs, cpu_count
from os.path import join
from glob import glob
from datetime import datetime

import numpy as np

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.metrics import Metric
from tensorflow.python.keras.callbacks import Callback, ModelCheckpoint, TensorBoard, LearningRateScheduler, EarlyStopping

from network import *




##################### Simple helper functions
def log_history(file, string, percent):
    file.write(string + '\n')
    print(string, end='\r' if percent < 100 else '\n')


def log_one_batch(phase, offset, total, loss, accuracy, history):
    percent = 100 * offset / total
    log_history(history,
                f"[{phase}: {percent:.3f}%] loss: {loss.result():.4f}, accuracy: {accuracy.result():.4f}",
                percent)
    if percent < 100: sys.stdout.flush()
    

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


def print_progress(prefix, value):
	"""print the progress status

   	# Arguments
   		prefix: the string that contains formatting typed string for print the progress status
    	value: the value that updated
	"""

	print(prefix.format(value), end='\r')
	sys.stdout.flush()


##################### model and checkpoint load functions
def load_model(model_name, reg, num_class, is_DPN=False):
	"""Load single model by model name

   	# arguments
   		model_name: string model name
   		reg: float of the regularization term (0.0 <= reg <= 1.0)
   		num_class: int of the number of class for classification
   		is_DPN: boolean that indicating that model is DPN

   	# returns
   		tf.keras.model.Model

	"""

	if model_name == "SRNet":
		model = Networks_structure_srnet.SRNet(reg, num_class, is_DPN)
		model.build(input_shape=(None, 256, 256, 1))
	elif model_name == "MISLNet":
		model = Networks_structure_mislnet.MISLNet(reg, num_class)
		model.build(input_shape=(None, 256, 256, 1))
	elif model_name == "DCTNet":
		model = Networks_structure_dctnet.DCTNet(reg, num_class, is_DPN)
		model.build(input_shape=(None, 256, 256, 1))
	elif model_name == "MesoNet":
		model = Networks_structure_mesonet.MesoNet(reg, num_class)
		model.build(input_shape=(None, 256, 256, 1))
	elif model_name == "XceptionNet":
		model = Networks_structure_xceptionnet.XceptionNet(reg, num_class)
		model.build(input_shape=(None, 256, 256, 1))
	elif model_name == "Fusion":
		model = Networks_structure_fusion.Fusion(reg, num_class)
		model.build(input_shape=(None, 1024))
	elif model_name == "Total":
		model = Networks_structure_total.Total(reg, num_class)
		model.build(input_shape=(None, 256, 256, 1))
	else:
		raise(BaseException("No such network: {}".format(NETWORK)))
	
	model.summary()
	return model


def load_logFiles(LOG_PATH, METHOD, NETWORK):
	"""Load log files data

	   	# arguments
	   		LOG_PATH: default log path
	   		METHOD: blur, median, noise, or multi
	   		NETWORK: the name of the network

	   	# returns
   		list of checkpoint path, history txt file, train summary writer, and valid summary writer

	"""
	# Set log file path
	current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
	log_path = join(LOG_PATH, current_time + "_{}_{}_".format(METHOD, NETWORK))
	ckpt_path = join(log_path, "checkpoint")
	makedirs(ckpt_path, exist_ok=True)
	history = open(join(log_path, 'log.txt'), 'w')

	# 2. tensorboard callback
	train_log_dir = join(log_path, 'gradient_tape', 'train')
	valid_log_dir = join(log_path, 'gradient_tape', 'validation')
	train_summary_writer = tf.summary.create_file_writer(train_log_dir)
	valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)

	return log_path, ckpt_path, history, train_summary_writer, valid_summary_writer


def load_ckpt(model, cktp_path, start_lr):
	"""Load checkpoint weights to the model

   	# arguments
   		model: tf.keras.model.Model
   		cktp_path: string checkpoint path
   		start_lr: float start learning rate of training model

	"""

	optimizer = tf.keras.optimizers.Adamax(lr=start_lr)
	loss = 'categorical_crossentropy'
	metrics = [tf.keras.metrics.CategoricalAccuracy()]

	model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

	if type(model) != Networks_structure_fusion.Fusion:
		model.build(input_shape=(None,256,256,1))
	else:
		model.build(input_shape=(None,1024))

	# model.summary()

	if cktp_path != "":
		model.load_weights(cktp_path)


def txt2list(txts):
	"""Read text files and returns the read lines

   	# arguments
   		txts: list of txt file paths

   	# returns
   		list of read lines

	"""

	fnames = []
	for txt in txts:
		with open(txt, 'r') as f:
			fnames += f.read().splitlines()

	return fnames



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
					tf.io.decode_raw(\
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
		'frames': tf.io.FixedLenFeature([], dtype=tf.string, default_value=""),
		'label'	: tf.io.FixedLenFeature([], dtype=tf.string, default_value=""),

		# additional information
		# 'br'	: tf.FixedLenFeature([], dtype=tf.int64, default_value=1)
	}

	# parse feature
	features = tf.io.parse_single_example(example_proto, feature_description)

	frames = _bytes_to_array(features, 'frames', tf.uint8, [256, 256, 3])
	frames = tf.image.rgb_to_grayscale(frames)

	label = _bytes_to_array(features, 'label', tf.uint8, [Networks_functions.NUM_CLASS])

	return frames, label


def configure_dataset(src_path, method, phase, batch_size, shuffle=True, repeat=True, frac=1.0):
	"""Configure the dataset 

   	# arguments
   		fnames: the list of strings of tfrecord file names
   		batch_size: the int represents batch size

	# Returns
		A dataset object configured by batch size
	"""
	fnames = txt2list(glob(join(src_path, method, f"{phase}_*.txt")))
	if shuffle: random.shuffle(fnames)
	fnames = fnames[:int(len(fnames) * frac)]
	buffer_size = max(len(fnames) / batch_size / 16, 16) # recommend buffer_size = # of elements / batches
	buffer_size = tf.cast(buffer_size, tf.int64)

	dataset = tf.data.TFRecordDataset(fnames)
	dataset = dataset.map(_parse_function, num_parallel_calls=cpu_count())
	dataset = dataset.prefetch(buffer_size=buffer_size)
	if shuffle: dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
	if repeat: dataset = dataset.repeat()
	dataset = dataset.batch(batch_size)

	return dataset, len(fnames) * 2


def configure_dataset_by_np(data_path, label_path, batch_size, shuffle=True):
	"""Configure the dataset by numpy file

   	# arguments
   		data_path: the path of numpy input data
   		label_path: the path of numpy input label
   		batch_size: the int represents batch size

	# Returns
		A dataset object configured by batch size
	"""

	data = np.load(data_path)
	label = np.load(label_path)

	idx = np.arange(data.shape[0])
	np.random.shuffle(idx)
	data = data[idx]
	label = label[idx]

	frac = 1.0
	data = data[:int(data.shape[0] * frac)]
	label = label[:int(label.shape[0] * frac)]

	buffer_size = max(data.shape[0] / batch_size, 16) # recommend buffer_size = # of elements / batches
	buffer_size = tf.cast(buffer_size, tf.int64)

	dataset = tf.data.Dataset.from_tensor_slices((data, label))
	dataset = dataset.prefetch(buffer_size=buffer_size) 
	if shuffle: dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True) 
	if shuffle: dataset = dataset.repeat()
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
		makedirs(training_log_dir, exist_ok=True)
		super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

	def set_model(self, model):
		if context.executing_eagerly():
			self.val_writer = tf.summary.create_file_writer(self.val_log_dir)
		else:
			self.val_writer = tf.summary.FileWriter(self.val_log_dir)
		super(TrainValTensorBoard, self).set_model(model)

	def _write_custom_summaries(self, step, logs=None):
		logs = logs or {}
		val_logs = {k.replace('val_', ''): v for k, v in logs.items() if 'val_' in k}
		if context.executing_eagerly():
			with self.val_writer.as_default(), tf.summary.always_record_summaries():
				for name, value in val_logs.items():
					tf.summary.scalar(name, value.item(), step=step)
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

	# NETWORK 			= args.network
	LOG_PATH 			= args.log_path
	METHOD 				= args.method
	BATCH_SIZE 			= args.batch_size
	LR_UPDATE_INTERVAL 	= args.lr_update_interval
	LR_UPDATE_RATE 		= args.lr_update_rate

	# Set log file path
	current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
	LOG_PATH = join(LOG_PATH, current_time + "_{}_{}_".format("all" if METHOD=="*" else METHOD, "NETWORK"))


	# 1. checkpoint callback
	ckpt_path = join(LOG_PATH, "checkpoint")
	makedirs(ckpt_path, exist_ok=True)
	ckpt_callback = SaveWeight(ckpt_path)
	

	# 2. tensorboard callback
	# tb_callback = TrainValTensorBoard(log_dir=LOG_PATH, update_freq='batch')
	tb_callback = TensorBoard(log_dir=LOG_PATH, update_freq='batch')


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
						# tb_callback, \
						lr_callback, \
						# earlyStop_callback, \
						# getWeight_callback, \
						]



##################### Custom metric objects
# Custum metric
class ConfusionMatrix(Metric):

	def __init__(self, name, **kwargs):
		super(ConfusionMatrix, self).__init__(name=name, num_class=2, **kwargs)
		self.confusion_matrix = self.add_weight(name='confusion_matrix', shape=(num_class, num_class), initializer='zeros')

	def update_state(self, y_true, y_pred, sample_weight=None):
		# self.confusion_matrix.assign_add(tf.ones((4,4)))
		# add = K.update_add(self.confusion_matrix, tf.ones((4,4)))
		y_true = tf.argmax(y_true, -1)[0]
		y_pred = tf.argmax(y_pred, -1)[0]
		
		operand = K.zeros_like(self.confusion_matrix)
		operand = operand.assign(K.zeros_like(self.confusion_matrix)[y_true, y_pred].assign(1.0))
		
		add = K.update_add(self.confusion_matrix, operand)
		
		# print1 = K.print_tensor(y_true, 'y_true: ')
		# print2 = K.print_tensor(y_pred, 'y_pred: ')
		# print3 = K.print_tensor(operand, 'operand: ')
		# print4 = K.print_tensor(self.confusion_matrix, 'confusion matrix: ')
		# ops = tf.group(add, print1, print2, print3, print4)
		ops = tf.group(add, print4)

		return ops

	def result(self):
		return tf.reduce_sum(self.confusion_matrix)

	def reset_states(self):
		# initial_matrix = tf.zeros((4,4))
		self.confusion_matrix.assign(K.zeros_like(self.confusion_matrix))

