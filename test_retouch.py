
import tensorflow as tf
import glob
import argparse
from os.path import join
from os import cpu_count

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
	parser.add_argument('--test_path', type=str, default='./retouch_tfrecord_test', help='test dataset path')
	parser.add_argument('--checkpoint_path', type=str, default='./logs/20190712_110608_blur/checkpoints/checkpoint_epoch_26_model.ckpt', help='checkpoint path')
	
	args = parser.parse_args()

	test_path = args.test_path
	test_fnames = glob.glob(join(test_path, "*", "*", "*.tfrecofd"))
	checkpoint_path = args.checkpoint_path


	# NUM_ELEMENT = 328880
	BATCH_SIZE = 1
	NUM_ELEMENT = len(test_fnames)
	buffer_size = tf.cast(NUM_ELEMENT / BATCH_SIZE / 16 / 16, tf.int64)
	buffer_size = 16

	files = tf.placeholder(dtype=tf.string)
	dataset = tf.data.TFRecordDataset(files)
	dataset = dataset.map(_parse_function, num_parallel_calls=cpu_count())
	dataset = dataset.batch(BATCH_SIZE)
	dataset = dataset.prefetch(buffer_size=buffer_size) # recommend buffer_size = # of elements / batches = 326,880 / 4 = 81,720
	dataset = dataset.shuffle(buffer_size=buffer_size) # recommend buffer_size = # of elements / batches
	dataset_iter = dataset.make_initializable_iterator()
	frames, label = dataset_iter.get_next() # (BATCH_SIZE * (3 * 256 * 256 * 3), batch_size * (4 * 1))
	# t = dataset_iter.get_next()


	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		saver = tf.train.import_meta_graph(checkpoint_path + ".meta")
		saver.restore(sess, checkpoint_path)
		print(sess.graph)

		FP = 0 # 오탐지
		FN = 0 # 미탐지
		test_step = 0
		sess.run(dataset_iter.initializer, feed_dict = {files: test_fnames})

		# while True:
		# 	try:
		# 		pass
		# 	except tf.errors.OutOfRangeError:
		# 		pass





















if __name__ == "__main__":
	main()


