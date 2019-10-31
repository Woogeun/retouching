"""
Util functions
@authorized by Shasha Bae
@description: util functions for printing writing training environments
"""

from network import *

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



##################### model and checkpoint load functions
def load_model(model_name, scale, reg = 0.0001, num_class):
	"""Load single model by model name

   	# arguments
   		model_name: string model name 
   		scale: float of the convolutional layer channel scale (0.0 < scale <= 1.0)
   		reg: float of the regularization term (0.0 <= reg <= 1.0)
   		num_class: int of the number of class for classification

   	# returns
   		tf.keras.model.Model

	"""

	if model_name == "SRNet":
		model = SRNet(scale, reg, num_class)
	elif model_name == "MISLNet":
		model = MISLNet(scale, reg, num_class)
	elif model_name == "DCTNet":
		model = DCTNet(scale, reg, num_class)
	elif model_name == "MesoNet":
		model = MesoNet(scale, reg, num_class)
	elif model_name == "Fusion":
		model = Fusion(scale, reg, num_class)
	else:
		raise(BaseException("No such network: {}".format(NETWORK)))

	return model


def load_cktp(model, cktp_path, start_lr):
	"""Load checkpoint weights to the model

   	# arguments
   		model: tf.keras.model.Model
   		cktp_path: string checkpoint path
   		start_lr: float start learning rate of training model

	"""
	
	optimizer = tf.keras.optimizers.Adamax(lr=start_lr)
	loss = 'categorical_crossentropy'
	metrics = {	"Accuracy": tf.keras.metrics.CategoricalAccuracy() }

	model.compile(optimizer=optimizer, loss=loss, metrics=list(metrics.values()))
	
	if type(model) != Fusion: 
		model.build(input_shape=(None,256,256,1))
	else:
		model.build(input_shape=(None,1024)) 

	model.summary()

	if not cktp_path: model.load_weights(cktp_path)


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

