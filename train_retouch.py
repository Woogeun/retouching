import argparse
from os.path import isdir, join, basename
from os import makedirs
import random
import datetime
import glob

import tensorflow as tf
import numpy as np
from network.Networks_structure_srnet import Network
# import cv2


def create_log_file(LOG_PATH, is_new=False):
	current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
	if is_new:
		LOG_PATH = join(LOG_PATH, current_time)

	CHECKPOINT_PATH = join(LOG_PATH, "checkpoints")
	if is_new:
		makedirs(CHECKPOINT_PATH)
	LOG_FOUT = open(join(LOG_PATH, "log.txt"), 'a')

	return LOG_PATH, CHECKPOINT_PATH, LOG_FOUT


def log_string(LOG_FOUT, out_str, is_print=True):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    if is_print:
    	print(out_str)


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
		# 'br'	: tf.FixedLenFeature([], dtype=tf.int64, default_value=1)
	}

	# parse feature
	features = tf.parse_single_example(example_proto, feature_description)

	frames = _bytes_to_array(features, 'frames', tf.uint8, [256, 256, 3])
	label = _bytes_to_array(features, 'label', tf.uint8, [2])

	# return frames, label, br
	return frames, label


def main():

	################################################## parse the arguments
	parser = argparse.ArgumentParser(description='Train retouch detection network.')
	parser.add_argument('--train_path', type=str, default='../retouch_tfrecord_train', help='train dataset path')
	parser.add_argument('--test_path', type=str, default='../retouch_tfrecord_test', help='test dataset path')
	parser.add_argument('--log_path', type=str, default='./logs', help='log path')
	parser.add_argument('--batch_size', type=int, default=4, help='batch size')
	parser.add_argument('--epoch', type=int, default=10, help='epoch')
	parser.add_argument('--summary_interval', type=int, default=100, help='loss inverval')
	parser.add_argument('--checkpoint_interval', type=int, default=100, help='checkpoint interval')
	parser.add_argument('--start_lr', type=float, default=1e-05, help='start learning rate')
	parser.add_argument('--lr_update_interval', type=int, default=10000, help='learning rate update interval')
	parser.add_argument('--lr_update_rate', type=float, default=0.95, help='learning rate update rate')
	args = parser.parse_args()

	method = "blur"
	br = 500


	# train directory validation check
	TRAIN_PATH = args.train_path
	if not isdir(TRAIN_PATH):
		raise(BaseException("no such directory \"%s\"" % TRAIN_PATH))
	train_fnames = glob.glob(join(TRAIN_PATH, method, "{}br", "*.tfrecord").format(br))
	random.shuffle(train_fnames)


	# test directory validation check
	TEST_PATH = args.test_path
	if not isdir(TEST_PATH):
		raise(BaseException("no such directory \"%s\"" % TEST_PATH))
	test_fnames = glob.glob(join(TEST_PATH, method, "{}br", "*.tfrecord").format(br))

	# create log directory with checkpoint
	LOG_PATH = args.log_path
	use_checkpoint = False

	log_files = glob.glob(join(LOG_PATH, "*"))

	if len(log_files) != 0: # there is least one train log file
		latest_log_path = log_files[-1]

		if basename(latest_log_path) == "log.txt": # LOG_PATH already indicating specific log file
			use_checkpoint = True

		else: # you can choose whether use lastest train checkpoint which is not done yet
			LOG_FOUT = open(join(latest_log_path, "log.txt"), 'r')
			for line in LOG_FOUT: pass
			LOG_FOUT.close()
			
			if line != "*********** End Training ***********\n":
				while True:
					answer = input("Use checkpoint? y or n ")
					if answer not in ['y', 'n']: continue
					
					if answer == 'y':
						use_checkpoint = True
						LOG_PATH = latest_log_path
					else:
						use_checkpoint = False
					break


	LOG_PATH, CHECKPOINT_PATH, LOG_FOUT = create_log_file(LOG_PATH, is_new=not use_checkpoint)
	# print("CHECKPOINT_PATH: {}".format(CHECKPOINT_PATH))


	# argument parse
	BATCH_SIZE = args.batch_size
	EPOCHS = args.epoch
	SUMMARY_INTERVAL = args.summary_interval
	CHECKPOINT_INTERVAL = args.checkpoint_interval
	START_LR = args.start_lr
	LR_UPDATE_INTERVAL = args.lr_update_interval
	LR_UPDATE_RATE = args.lr_update_rate





	############################################### load dataset
	# total 240(4 videos * 15 clips * 4 attacks) elements in sample data
	files = tf.placeholder(dtype=tf.string)
	dataset = tf.data.TFRecordDataset(files)
	dataset = dataset.map(_parse_function, num_parallel_calls=8)
	dataset = dataset.batch(BATCH_SIZE)
	dataset = dataset.prefetch(buffer_size=1000) # prefetch (BATCH_SIZE * buffer_size)
	dataset = dataset.shuffle(buffer_size=1000)
	dataset_iter = dataset.make_initializable_iterator()
	frames, label = dataset_iter.get_next() # (BATCH_SIZE * (3 * 256 * 256 * 3), batch_size * (4 * 1))
	# t = dataset_iter.get_next()
	



	############################################### training
	with tf.Session() as sess:

		# # input value check
		# sess.run([dataset_iter.initializer], feed_dict={files: train_fnames})
		# while True:
		# 	try:
		# 		t_ = sess.run([t])
		# 		print(t_[0][1])
		# 	except tf.errors.OutOfRangeError:
		# 		break


		# learning rate setup
		global_step = tf.Variable(0, trainable=False)
		learning_rate = tf.train.exponential_decay(START_LR, global_step, LR_UPDATE_INTERVAL, LR_UPDATE_RATE, staircase=True)

		# network setup
		phase = tf.placeholder(tf.bool, name='phase')
		loss, accuracy, merge = Network(frames, label, phase)

		# train operation with Adam optimizer
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops): # update moving mean and moving variance during training
			train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

		# tensorboard writer
		writer = tf.summary.FileWriter(LOG_PATH, sess.graph)

		# checkpoint saver
		if False:
			saver = tf.train.import_meta_graph(join(CHECKPOINT_PATH, "checkpoint_epoch_0_model.ckpt.meta"))
		else:
			variables = tf.global_variables()
			saver = tf.train.Saver(var_list=variables, write_version=tf.train.SaverDef.V2)
		
		
		# train
		log_string(LOG_FOUT, "*********** Start Training ***********")
		tf.global_variables_initializer().run()

		if use_checkpoint:
			ckpt_path = saver.restore(sess, tf.train.latest_checkpoint(CHECKPOINT_PATH))
			print("checkpoint is restored from \"{}\"".format(ckpt_path))
		total_step = 1

		for epoch in range(EPOCHS):
			
			# train session(one epoch)
			log_string(LOG_FOUT, "*********** Train EPOCH {} ***********".format(epoch+1))
			sess.run(dataset_iter.initializer, feed_dict = {files: train_fnames})
			train_losses = 0
			train_step = 1
			
			while True:
				try:
					_, _loss, summary = sess.run([train_op, loss, merge], feed_dict={phase: True})
					train_losses += _loss
					total_step += 1
					train_step += 1

					if total_step % SUMMARY_INTERVAL == 0:
						log_string(LOG_FOUT, "{} interation | loss : {}".format(total_step, _loss))
						writer.add_summary(summary, total_step)

				except tf.errors.OutOfRangeError:
					# log_string(LOG_FOUT, "{} interation | loss : {}".format(total_step, _loss))
					saver.save(sess, join(CHECKPOINT_PATH, "checkpoint_epoch_{}_model.ckpt").format(epoch))
					break

			average_loss = train_losses / train_step


			# test session
			sess.run(dataset_iter.initializer, feed_dict = {files: test_fnames})
			acc_test = 0
			test_step = 1
			while True:
				try:
					_acc = sess.run([accuracy], feed_dict={phase:False})[0]
					acc_test += _acc
					test_step += 1

				except tf.errors.OutOfRangeError:
					break

			acc_test /= test_step
			log_string(LOG_FOUT, "Test average loss: {:.5f}, accuracy: {:.5f}".format(average_loss, acc_test))


		log_string(LOG_FOUT, "*********** End Training ***********")


	LOG_FOUT.close()









if __name__ == "__main__":
	main()





