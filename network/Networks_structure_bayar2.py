#Describe networks structure
#BayarNet
from Networks_functions import *

def bayar_conv(input):
	#Bayer filter
	w_temp = 0.1 * np.random.randn(5,5,1,3)
	w_nor = np.zeros([5,5,1,3])
	k_sum = np.zeros(3)
	w_temp[2,2,0,:] = 0
	for k in range(0,3):
		for i in range(0,5):
			for j in range(0,5):
				k_sum[k] = k_sum[k] + w_temp[i,j,0,k]
	for k in range(0,3):
		w_nor[:,:,:,k] = np.divide(w_temp[:,:,:,k],k_sum[k])
	w_nor[2,2,0,:] = -1
	w_new = tf.Variable(tf.constant(w_nor.tolist()))
	return conv2d(input,w_new)

#input and labels
x = tf.placeholder(tf.float32, shape=[None,256,256,3])
y_ = tf.placeholder(tf.float32, shape=[None, 2])
phase = tf.placeholder(tf.bool, name='phase')

x = tf.image.rgb_to_yuv(x) #output : [0~1]
x_y = x[:,:,:,0]
x_y = tf.reshape(x_y,[-1,256,256,1])

#0 Constrained Conv layer
x_y_bayar = bayar_conv(x_y)

#1 Conv layer
w1 = weight_variable([7,7,3,96])
h1 = TanH(batch_norm(conv2d_2(x_y_bayar,w1),phase))
p1 = max_pool_3x3_2(h1)

#2 Conv layer
w2 = weight_variable([5,5,96,64])
h2 = TanH(batch_norm(conv2d(p1,w2),phase))
p2 = max_pool_3x3_2(h2)

#3 Conv layer
w3 = weight_variable([5,5,64,64])
h3 = TanH(batch_norm(conv2d(p2,w3),phase))
p3 = max_pool_3x3_2(h3)

#4 Conv layer
w4 = weight_variable([1,1,64,128])
h4 = TanH(batch_norm(conv2d(p3,w4),phase))
p4 = avg_pool_3x3_2(h4)

p4_flat = tf.reshape(p4,[-1,8*8*128])

#Fully connected layer variable 1
w_fc1 = weight_variable([8*8*128,200])
b_fc1 = bias_variable([200])
h_fc1 = FC(p4_flat,w_fc1) + b_fc1

#Fully connected layer variable 2
w_fc2 = weight_variable([200,200])
b_fc2 = bias_variable([200])
h_fc2 = FC(h_fc1,w_fc2) + b_fc2

#Fully connected layer variable 3
w_fc3 = weight_variable([200,2])
b_fc3 = bias_variable([2])
y = FC(h_fc2,w_fc3) + b_fc3
softmax_y = tf.nn.softmax(y)
# print y

# W_L2 = l2(w1) + l2(w1_2) + l2(w2) + l2(w2_2) + l2(w3) + l2(w3_2) + l2(w3_3) + l2(w4) + l2(w4_2) + l2(w4_3) + l2(w5) + l2(w5_2) + l2(w5_3) + l2(w_fc1) + l2(w_fc2) + l2(w_fc3)
W_L2 = 0

