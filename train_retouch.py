import argparse
from os.path import isdir, join, basename
from os import makedirs, cpu_count
import random
import datetime
import glob

import tensorflow as tf
import numpy as np
from network.Networks_structure_srnet import Network



def create_log_file(LOG_PATH, is_new=False):
	current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
	if is_new:
		LOG_PATH = join(LOG_PATH, current_time)

	CHECKPOINT_PATH = join(LOG_PATH, "checkpoints")
	LOG_PATH_TRAIN = join(LOG_PATH, "train")
	LOG_PATH_TEST = join(LOG_PATH, "test")
	if is_new:
		makedirs(CHECKPOINT_PATH)
		makedirs(LOG_PATH_TRAIN)
		makedirs(LOG_PATH_TEST)
	LOG_FOUT_TRAIN = open(join(LOG_PATH_TRAIN, "log_train.txt"), 'a')
	LOG_FOUT_TEST = open(join(LOG_PATH_TEST, "log_test.txt"), 'a')

	return LOG_PATH_TRAIN, LOG_PATH_TEST, CHECKPOINT_PATH, LOG_FOUT_TRAIN, LOG_FOUT_TEST


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
	parser.add_argument('--train_path', type=str, default='./retouch_tfrecord_train', help='train dataset path')
	parser.add_argument('--test_path', type=str, default='./retouch_tfrecord_test', help='test dataset path')
	parser.add_argument('--method', type=str, default="blur", help='attack method')
	parser.add_argument('--br', type=str, default="500", help='bitrate')
	parser.add_argument('--log_path', type=str, default='./logs', help='log path')
	parser.add_argument('--batch_size', type=int, default=16, help='batch size')
	parser.add_argument('--epoch', type=int, default=3, help='epoch')
	parser.add_argument('--summary_interval', type=int, default=1, help='loss inverval')
	parser.add_argument('--checkpoint_interval', type=int, default=500, help='checkpoint interval')
	parser.add_argument('--start_lr', type=float, default=1e-05, help='start learning rate')
	parser.add_argument('--lr_update_interval', type=int, default=10000, help='learning rate update interval')
	parser.add_argument('--lr_update_rate', type=float, default=0.95, help='learning rate update rate')
	
	args = parser.parse_args()

	method = args.method
	br = args.br


	# train directory validation check
	TRAIN_PATH = args.train_path
	if not isdir(TRAIN_PATH):
		raise(BaseException("no such directory \"%s\"" % TRAIN_PATH))
	train_fnames = glob.glob(join(TRAIN_PATH, method, "{}br", "*.tfrecord").format(br))
	random.shuffle(train_fnames)
	# train_fnames = train_fnames[:len(train_fnames) // 3]


	# test directory validation check
	TEST_PATH = args.test_path
	if not isdir(TEST_PATH):
		raise(BaseException("no such directory \"%s\"" % TEST_PATH))
	test_fnames = glob.glob(join(TEST_PATH, method, "{}br", "*.tfrecord").format(br))
	# test_fnames = test_fnames[:len(test_fnames) // 3]

	# create log directory with checkpoint
	LOG_PATH = args.log_path
	use_checkpoint = False

	log_files = glob.glob(join(LOG_PATH, "*"))

	if False:
	# if len(log_files) != 0: # there is least one train log file
		latest_log_path = log_files[-1]

		if basename(latest_log_path) == "log.txt": # LOG_PATH already indicating specific log file
			use_checkpoint = True

		else: # you can choose whether use lastest train checkpoint which is not done yet
			LOG_FOUT_TRAIN = open(join(latest_log_path, "log_train.txt"), 'r')
			line = None
			for line in LOG_FOUT_TRAIN: pass
			LOG_FOUT_TRAIN.close()
			
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


	LOG_PATH_TRAIN, LOG_PATH_TEST, CHECKPOINT_PATH, LOG_FOUT_TRAIN, LOG_FOUT_TEST = create_log_file(LOG_PATH, is_new=not use_checkpoint)
	log_string(LOG_FOUT_TRAIN, str(vars(args)))
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
	# NUM_ELEMENT = 328880
	NUM_ELEMENT = len(train_fnames)
	buffer_size = tf.cast(NUM_ELEMENT / BATCH_SIZE / 16 / 16, tf.int64)
	buffer_size = 16

	files = tf.placeholder(dtype=tf.string)
	dataset = tf.data.TFRecordDataset(files)
	# dataset = dataset.apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=cpu_count()))
	dataset = dataset.map(_parse_function, num_parallel_calls=cpu_count())
	dataset = dataset.batch(BATCH_SIZE)
	dataset = dataset.prefetch(buffer_size=buffer_size) # recommend buffer_size = # of elements / batches = 326,880 / 4 = 81,720
	dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True) # recommend buffer_size = # of elements / batches
	# dataset = dataset.repeat(EPOCHS)
	dataset_iter = dataset.make_initializable_iterator()
	frames, label = dataset_iter.get_next() # (BATCH_SIZE * (3 * 256 * 256 * 3), batch_size * (4 * 1))
	# t = dataset_iter.get_next()
	
	# print(NUM_ELEMENT)


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
		loss, accuracy = Network(frames, label, phase)

		# train operation with Adam optimizer
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops): # update moving mean and moving variance during training
			train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

		# tensorboard writer
		writer_train = tf.summary.FileWriter(LOG_PATH_TRAIN, sess.graph)
		writer_test = tf.summary.FileWriter(LOG_PATH_TEST)

		# checkpoint saver
		if False:
			saver = tf.train.import_meta_graph(join(CHECKPOINT_PATH, "checkpoint_epoch_0_model.ckpt.meta"))
		else:
			variables = tf.global_variables()
			saver = tf.train.Saver(var_list=variables, write_version=tf.train.SaverDef.V2, max_to_keep=None)
		
		
		# train
		log_string(LOG_FOUT_TRAIN, "*********** Start Training ***********")
		tf.global_variables_initializer().run()
		# sess.run(tf.contrib.layers.variance_scaling_initializer())

		if use_checkpoint:
			ckpt_path = saver.restore(sess, tf.train.latest_checkpoint(CHECKPOINT_PATH))
			print("checkpoint is restored from \"{}\"".format(ckpt_path))
		total_step = 0

		for epoch in range(EPOCHS):
			
			# train session(one epoch)
			log_string(LOG_FOUT_TRAIN, "*********** Train EPOCH {} ***********".format(epoch+1))
			sess.run(dataset_iter.initializer, feed_dict = {files: train_fnames})
			train_avg_loss = 0
			train_avg_acc = 0
			train_step = 0
			
			while True:
				try:
					_, train_acc, train_loss = sess.run([train_op, accuracy, loss], feed_dict={phase: True})
					train_avg_loss += train_loss
					train_avg_acc += train_acc
					train_step += 1
					total_step += 1
					

					# if True:
					if total_step % SUMMARY_INTERVAL == 0:
						# summary files
						train_avg_loss /= train_step
						train_avg_acc /= train_step

						train_summary = tf.Summary()
						train_summary.value.add(tag="train/loss", simple_value=train_avg_loss)
						# train_summary.value.add(tag="loss/train", simple_value=train_avg_loss)
						train_summary.value.add(tag="train/accuracy", simple_value=train_avg_acc)
						# train_summary.value.add(tag="accuracy/train", simple_value=train_avg_acc)

						# tf.summary.scalar('loss', tf.cast(train_avg_loss, tf.float32))
						# tf.summary.scalar('accuracy', tf.cast(train_avg_acc, tf.float32))
						# train_summary = tf.summary.merge_all()

						writer_train.add_summary(train_summary, total_step)
						log_string(LOG_FOUT_TRAIN, "{} interation | loss : {}".format(total_step, train_loss))

						train_avg_loss = 0
						train_avg_acc = 0
						train_step = 0
						

				except tf.errors.OutOfRangeError:
					saver.save(sess, join(CHECKPOINT_PATH, "checkpoint_epoch_{}_model.ckpt").format(epoch))
					break
					

			# test session
			sess.run(dataset_iter.initializer, feed_dict = {files: test_fnames})
			test_avg_acc = 0
			test_avg_loss = 0
			test_step = 0

			while True:
				try:
					test_acc, test_loss = sess.run([accuracy, loss], feed_dict={phase:False})
					test_avg_acc += test_acc
					test_avg_loss += test_loss
					test_step += 1

				except tf.errors.OutOfRangeError:
					break

			# summary files
			test_avg_acc /= test_step
			test_avg_loss /= test_step

			test_summary = tf.Summary()
			test_summary.value.add(tag="test/loss", simple_value=test_avg_loss)
			# test_summary.value.add(tag="loss/test", simple_value=test_avg_loss)
			test_summary.value.add(tag="test/accuracy", simple_value=test_avg_acc)
			# test_summary.value.add(tag="accuracy/test", simple_value=test_avg_acc)

			# tf.summary.scalar('loss', tf.cast(test_avg_loss, tf.float32))
			# tf.summary.scalar('accuracy', tf.cast(test_avg_acc, tf.float32))
			# test_summary = tf.summary.merge_all()

			writer_test.add_summary(test_summary, total_step)
			log_string(LOG_FOUT_TEST, "Test average loss: {:.5f}, accuracy: {:.5f}".format(test_avg_loss, test_avg_acc))

			


		log_string(LOG_FOUT_TRAIN, "*********** End Training ***********")


	LOG_FOUT_TRAIN.close()
	LOG_FOUT_TEST.close()









if __name__ == "__main__":
	main()





