#Describe networks structure

from .Networks_functions import *

def layer_type1(layer_input, in_dimension, out_dimension, phase):
	t1_weights = weight_variable([3,3,in_dimension,out_dimension])
	layer_output = ReLU(batch_norm(conv2d(layer_input, t1_weights), phase))
	return layer_output

def layer_type2(layer_input, in_out_dimension, phase):
	t2_weights = weight_variable([3,3,in_out_dimension,in_out_dimension])
	t2_2_weights = weight_variable([3, 3, in_out_dimension, in_out_dimension])
	y1 = ReLU(batch_norm(conv2d(layer_input, t2_weights), phase))
	y2 = batch_norm(conv2d(y1, t2_2_weights), phase)
	layer_output = layer_input + y2
	return layer_output

def layer_type3(layer_input, in_dimension,out_dimension, phase):
	t3_weights = weight_variable([3,3,in_dimension,out_dimension])
	t3_2_weights = weight_variable([3,3,out_dimension,out_dimension])
	t3_3_weights = weight_variable([1,1,in_dimension,out_dimension])
	y1 = ReLU(batch_norm(conv2d(layer_input, t3_weights), phase))
	y2 = batch_norm(conv2d(y1, t3_2_weights), phase)
	y3 = tf.nn.avg_pool(y2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
	y_left = conv2d_2(layer_input,t3_3_weights)
	layer_output = y3 + y_left
	return layer_output

def layer_type4(layer_input, hw_size, in_dimension, out_dimension, phase):
	t4_weights = weight_variable([3,3,in_dimension,out_dimension])
	t4_2_weights = weight_variable([3, 3, out_dimension, out_dimension])
	y1 = ReLU(batch_norm(conv2d(layer_input, t4_weights), phase))
	y2 = batch_norm(conv2d(y1, t4_2_weights), phase)
	h, w = hw_size
	layer_output = tf.nn.avg_pool(y2, ksize=[1, h, w, 1], strides=[1, 1, 1, 1], padding="VALID")
	#layer_output is a vector (out_dimension)
	return layer_output

def Network(frames, label, phase):
	#input and labels
	# frames = tf.placeholder(tf.float32, shape=[None,256,256,3])
	# label = tf.placeholder(tf.float32, shape=[None, 2])
	# QT = tf.placeholder(tf.float32, shape=[None, 8, 8])

	#RGB to YCbCr
	yuv = tf.image.rgb_to_yuv(frames)
	x1 = yuv[:, :, :, 0]
	x1 = tf.reshape(x1, [-1, 256, 256, 1])

	#Convnet
	# channel_scale = 0.4
	channel_scale = 1

	#input Nx256x256x1
	L1_T1 = layer_type1(x1,1,int(64*channel_scale), phase)
	L2_T1 = layer_type1(L1_T1,int(64*channel_scale),int(16*channel_scale), phase)

	L3_T2 = layer_type2(L2_T1,int(16*channel_scale), phase)
	L4_T2 = layer_type2(L3_T2,int(16*channel_scale), phase)
	L5_T2 = layer_type2(L4_T2,int(16*channel_scale), phase)
	L6_T2 = layer_type2(L5_T2,int(16*channel_scale), phase)
	L7_T2 = layer_type2(L6_T2,int(16*channel_scale), phase)

	L8_T3 = layer_type3(L7_T2,int(16*channel_scale),int(16*channel_scale), phase) #128x128
	L9_T3 = layer_type3(L8_T3,int(16*channel_scale),int(64*channel_scale), phase) #64x64
	L10_T3 = layer_type3(L9_T3,int(64*channel_scale),int(128*channel_scale), phase) #32x32
	L11_T3 = layer_type3(L10_T3,int(128*channel_scale),int(256*channel_scale), phase) #16x16
	# print(L11_T3)

	L12_T4 = layer_type4(L11_T3,[16,16],int(256*channel_scale),int(512*channel_scale), phase)
	# print(L12_T4)
	p3_flat = tf.reshape(L12_T4,[-1,int(512*channel_scale)])

	#Fully connected layer variable 3
	w_fc3 = weight_variable([int(512*channel_scale),2])
	b_fc3 = bias_variable([2])
	y = FC(p3_flat,w_fc3)# + b_fc3
	y_softmax = tf.nn.softmax(y)
	'''
	#W_L2 = l2(w1) + l2(b1) + l2(w2) + l2(b2) + l2(w_fc1) + l2(b_fc1) + l2(w_fc2) + l2(b_fc2) + l2(w_fc3) + l2(b_fc3)
	'''
	W_L2 = 0

	beta = 0.001
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=label))
	loss = cross_entropy + beta * W_L2

	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(label,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

	# summary files
	tf.summary.scalar('train/cross_entropy', tf.reduce_mean(cross_entropy))
	tf.summary.scalar('train/accuracy', tf.reduce_mean(accuracy))
	merge = tf.summary.merge_all()

	return loss, accuracy, merge
