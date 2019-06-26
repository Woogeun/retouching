import argparse
from os.path import isdir, join
from os import makedirs
import random
import datetime
import glob

import tensorflow as tf
import numpy as np
# import cv2


def _bytes_to_array(features, key, element_type, dimension):
	return 	tf.cast(\
				tf.reshape(\
					tf.decode_raw(\
						features[key],\
						element_type),\
					dimension) ,\
				tf.float32)


# configure tf.Example proto data as dictionary
def _parse_function(example_proto):
	feature_description = {
		# mendatory informations
		'frames': tf.FixedLenFeature([], dtype=tf.string, default_value=""),
		'label'	: tf.FixedLenFeature([], dtype=tf.string, default_value=""),

		# additional information
		# 'fps'	: tf.FixedLenFeature([], dtype=tf.int64, default_value=1)
	}

	# parse feature
	features = tf.parse_single_example(example_proto, feature_description)

	frames = _bytes_to_array(features, 'frames', tf.uint8, [3, 256, 256, 3])
	label = _bytes_to_array(features, 'label', tf.uint8, [4, 1])
	# fps = tf.cast(features['fps'], tf.int64)

	# return frames, label, fps
	return frames, label



def main():

	################################################## parse the arguments
	parser = argparse.ArgumentParser(description='Train retouch detection network.')
	parser.add_argument('--train_path', type=str, default='./retouch_tfrecord_train', help='train dataset path')
	parser.add_argument('--test_path', type=str, default='./retouch_tfrecord_test', help='test dataset path')
	parser.add_argument('--log_path', type=str, default='./logs', help='checkpoint path')
	parser.add_argument('--batch_size', type=int, default=32, help='batch size')
	parser.add_argument('--epoch', type=int, default=3, help='epoch')
	parser.add_argument('--loss_inverval', type=int, default=100, help='loss inverval')
	parser.add_argument('--save_interval', type=int, default=100, help='checkpoint interval')
	parser.add_argument('--start_lr', type=float, default=1e-05, help='start learning rate')
	parser.add_argument('--lr_update_interval', type=int, default=10000, help='learning rate update interval')
	parser.add_argument('--lr_update_rate', type=float, default=0.95, help='learning rate update rate')
	args = parser.parse_args()

	# train directory validation check
	train_path = args.train_path
	if not isdir(train_path):
		raise(BaseException("no such directory \"%s\"" % train_path))
	train_fnames = glob.glob(join(train_path, "*", "{}fps", "*.tfrecord").format("*"))
	random.shuffle(train_fnames)

	# test directory validation check
	test_path = args.test_path
	if not isdir(test_path):
		raise(BaseException("no such directory \"%s\"" % test_path))
	test_fnames = glob.glob(join(test_path, "*", "{}fps", "*.tfrecord").format("*"))

	# create log directory
	log_path = args.log_path
	current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	log_path = join(log_path, current_time)
	makedirs(join(log_path, "train"))
	makedirs(join(log_path, "test"))

	# argument parse
	batch_size = args.batch_size
	epoch_total = args.epoch
	loss_inverval = args.loss_inverval
	save_interval = args.save_interval
	start_lr = args.start_lr
	lr_update_interval = args.lr_update_interval
	lr_update_rate = args.lr_update_rate



	############################################### load dataset
	# total 240(15 * 4 * 4) elements in sample data
	files = tf.placeholder(dtype=tf.string)
	dataset = tf.data.TFRecordDataset(files)
	dataset = dataset.map(_parse_function, num_parallel_calls=1)
	dataset = dataset.batch(batch_size)
	dataset = dataset.shuffle(buffer_size=1000)
	dataset = dataset.prefetch(buffer_size=1000) # prefetch (batch_size * buffer_size)
	dataset_iter = dataset.make_initializable_iterator()
	# frames, label, fps = dataset_iter.get_next()
	next_element = dataset_iter.get_next() # (batch_size * (3 * 256 * 256 * 3), batch_size * (4 * 1))

	
	with tf.Session() as sess:

		# train one epoch
		sess.run(dataset_iter.initializer, feed_dict = {files: train_fnames})
		while True:
			try:
				t = sess.run(next_element)[1]
				# print(t)
				# print(np.average(t, axis=0))
			except tf.errors.OutOfRangeError:
				print("********************End of train********************")
				break











if __name__ == "__main__":
	main()





