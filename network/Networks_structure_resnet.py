#Describe networks structure

from Networks_functions import *

def resi_layer(re_input, dimension):
	c_weights = weight_variable([3,3,dimension,dimension])
	c2_weights = weight_variable([3,3,dimension,dimension])
	re_y1 = ReLU(batch_norm(conv2d(re_input, c_weights),phase))
	re_y2 = batch_norm(conv2d(re_y1, c2_weights),phase)
	re_output = ReLU(re_y2 + re_input)

	return re_output

def resi_layer_down(re_input, f_dimension, s_dimension):
#stride 2
	c_weights = weight_variable([3,3,f_dimension,s_dimension])
	c2_weights = weight_variable([3,3,s_dimension,s_dimension])
	re_y1 = ReLU(batch_norm(conv2d_2(re_input, c_weights),phase))
	re_y2 = batch_norm(conv2d(re_y1, c2_weights),phase)

	c3_weights = weight_variable([1,1,f_dimension,s_dimension])
	re_input_down = batch_norm(conv2d_2(re_input, c3_weights),phase)

	re_output = ReLU(re_y2 + re_input_down)

	return re_output	

#input and labels
x = tf.placeholder(tf.float32, shape=[None,256,256,3])
y_ = tf.placeholder(tf.float32, shape=[None, 2])
# QT = tf.placeholder(tf.float32, shape=[None, 8, 8])
phase = tf.placeholder(tf.bool, name='phase')

#rgb to ycbcr
yuv = tf.image.rgb_to_yuv(x)
x1 = yuv[:, :, :, 0]
x1 = tf.reshape(x1, [-1,256,256,1])

#Convnet
C = 64
#input Nx128x128x3
w1 = weight_variable([7,7,1,C])
h1 = ReLU(batch_norm(conv2d_2(x1,w1),phase))
#p1 = max_pool_2x2(h1)

#Nx64x64xC
re1_1 = resi_layer(h1, C)
re1_2 = resi_layer(re1_1, C)
re1_3 = resi_layer(re1_2, C)

#Nx64x64xC
re2_1 = resi_layer_down(re1_3, C, 2*C)
#Nx32x32x2C
re2_2 = resi_layer(re2_1, 2*C)
re2_3 = resi_layer(re2_2, 2*C)
re2_4 = resi_layer(re2_3, 2*C)

re3_1 = resi_layer_down(re2_4, C*2, 2*C)
#Nx32x32x2C
re3_2 = resi_layer(re3_1, 2*C)
re3_3 = resi_layer(re3_2, 2*C)
re3_4 = resi_layer(re3_3, 2*C)

#Nx32x32x2C
re4_1 = resi_layer_down(re3_4, 2*C, 4*C)
#Nx16x16x4C
re4_2 = resi_layer(re4_1, 4*C)
re4_3 = resi_layer(re4_2, 4*C)
re4_4 = resi_layer(re4_3, 4*C)
re4_5 = resi_layer(re4_4, 4*C)
re4_6 = resi_layer(re4_5, 4*C)

#Nx16x16x4C
re5_1 = resi_layer_down(re4_6, 4*C, 8*C)
#Nx8x8x8C
re5_2 = resi_layer(re5_1, 8*C)
re5_3 = resi_layer(re5_2, 8*C)

p3 = avg_pool_2x2(re5_3)
p3_flat = tf.reshape(p3,[-1,4*4*8*C])

#Fully connected layer variable 3
w_fc3 = weight_variable([4*4*8*C,2])
b_fc3 = bias_variable([2])
y = FC(p3_flat,w_fc3) + b_fc3
y_softmax = tf.nn.softmax(y)
'''
#W_L2 = l2(w1) + l2(b1) + l2(w2) + l2(b2) + l2(w_fc1) + l2(b_fc1) + l2(w_fc2) + l2(b_fc2) + l2(w_fc3) + l2(b_fc3)
'''
W_L2 = 0
