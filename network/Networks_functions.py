import tensorflow as tf
import numpy as np
from scipy.fftpack import dct,idct
from scipy.signal import convolve2d
from PIL import Image
import math

#caclulate DCT basis
def cal_scale(p,q): #for 8x8 dct
	if p==0:
		ap = 1/(math.sqrt(8))
	else:
		ap = math.sqrt(0.25) #0.25 = 2/8
	if q==0:
		aq = 1/(math.sqrt(8))
	else:
		aq = math.sqrt(0.25) #0.25 = 2/8

	return ap,aq

def cal_basis(p,q): #for 8x8 dct
	basis = np.zeros((8,8))
	ap,aq = cal_scale(p,q)
	for m in range(0,8):
		for n in range(0,8):
			basis[m,n] = ap*aq*math.cos(math.pi*(2*m+1)*p/16)*math.cos(math.pi*(2*n+1)*q/16)

	return basis

def load_DCT_basis_64(): #for 8x8 dct
	basis_64 = np.zeros((8,8,64))
	idx = 0
	for i in range(8):
		for j in range(8):
			basis_64[:,:,idx] = cal_basis(i,j)
			idx = idx + 1
	return basis_64

def cal_scale_16(p,q):
	if p==0:
		ap = 1/(math.sqrt(4))
	else:
		ap = math.sqrt(0.5) #0.5 = 2/4
	if q==0:
		aq = 1/(math.sqrt(4))
	else:
		aq = math.sqrt(0.5) #0.5 = 2/4
	return ap,aq

def cal_basis_16(p,q):
	basis = np.zeros((4,4))
	ap,aq = cal_scale_16(p,q)
	for m in range(0,4):
		for n in range(0,4):
			basis[m,n] = ap*aq*math.cos(math.pi*(2*m+1)*p/8)*math.cos(math.pi*(2*n+1)*q/8)
	return basis


def load_DCT_basis_16():
	basis_16 = np.zeros((4,4,16))
	idx = 0
	for i in range(4):
		for j in range(4):
			basis_16[:,:,idx] = cal_basis_16(i,j)
			idx = idx + 1
	return basis_16

def cal_scale_256(p,q):
	if p==0:
		ap = 1/(math.sqrt(16))
	else:
		ap = math.sqrt(0.125) #0.125 = 2/16
	if q==0:
		aq = 1/(math.sqrt(16))
	else:
		aq = math.sqrt(0.125) #0.125 = 2/16
	return ap,aq

def cal_basis_256(p,q):
	basis = np.zeros((16,16))
	ap,aq = cal_scale_256(p,q)
	for m in range(0,16):
		for n in range(0,16):
			basis[m,n] = ap*aq*math.cos(math.pi*(2*m+1)*p/32)*math.cos(math.pi*(2*n+1)*q/32)
	return basis


def load_DCT_basis_256():
	basis_256 = np.zeros((16,16,256))
	idx = 0
	for i in range(16):
		for j in range(16):
			basis_256[:,:,idx] = cal_basis_256(i,j)
			idx = idx + 1
	return basis_256


#networks functions
def l2(x):
	return tf.nn.l2_loss(x)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def batch_norm(x,phase):
	return tf.contrib.layers.batch_norm(x, decay=0.9, epsilon=0.001, center=True, scale=True, is_training=phase)

def conv2d_v(x,W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="VALID")

def conv2d(x,W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

def conv2d_2(x,W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding="SAME")

def conv2d_16(x,W):
    return tf.nn.conv2d(x, W, strides=[1, 16, 16, 1], padding="SAME")
	
def conv2d_8(x,W):
    return tf.nn.conv2d(x, W, strides=[1, 8, 8, 1], padding="SAME")

def conv2d_4(x,W):
    return tf.nn.conv2d(x, W, strides=[1, 4, 4, 1], padding="SAME")

def conv2d_k(x,W,k):
    return tf.nn.conv2d(x, W, strides=[1,k,k,1], padding="SAME")

def conv3d(x,W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding="SAME")

def max_pool_3d_3x3(x):
    return tf.nn.max_pool3d(x, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding="SAME")

def max_pool_3d_2x2(x):
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def max_pool_3x3_2(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

def avg_pool_3x3_2(x):
    return tf.nn.avg_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

def max_pool_3x1_2_v(x):
	return tf.nn.max_pool(x, ksize=[1, 3, 1, 1], strides=[1, 2, 1, 1], padding="VALID")

def avg_pool_2x2(x):
	return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def avg_pool_5x5(x):
	return tf.nn.avg_pool(x, ksize=[1, 5, 5, 1], strides=[1, 2, 2, 1], padding="SAME")

def avg_pool_16x16_v(x):
	return tf.nn.avg_pool(x, ksize=[1, 16, 16, 1], strides=[1, 1, 1, 1], padding="VALID")

def FC(x,W):
	return tf.matmul(x,W)

def ReLU(x):
    return tf.nn.relu(x)

def TanH(x):
    return tf.nn.tanh(x)
