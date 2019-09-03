#18-10-01 written by Jinseok Park
#This code convert single jpeg images to h5 file that contains manipulated double images

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import random
import io
import sys

'''
Information of blocks
blocks - tuple of size 2
blocks[0] - set of blocks (50,256,256,3) 
blocks[1] - set of labels (50,2), 
original label - blocks[1][:,0] = 1, blocks[1][:,1] = 0
forged label - blocks[1][:,0] = 0, blocks[1][:,1] = 1
'''

# Shuffle in unison
def unison_shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p],b[p]

def delete_outrange(np_img):
    L = np_img>255
    M = np_img<0

    np_img[L] = 255
    np_img[M] = 0

    np_img = np_img.astype('uint8')
    return np_img

def jpeg_compression(img,quality_factor):
    im = Image.fromarray(img)
    temp_path = "./temp.jpg"
    quality_factor = int(quality_factor)

    if quality_factor == -1:
        rq = random.randrange(1,12) # 1~11
        quality_factor = 45 + rq*5 # quality factor : 50~100
    elif quality_factor == -2:
        rq = random.randrange(0,51) # 0~50
        quality_factor = 50 + rq

    im.save(temp_path,'JPEG',quality=quality_factor)

    #read image
    jpeg_im = Image.open(temp_path).convert('RGB')
    return np.array(jpeg_im, dtype='uint8')


def jpeg_compression_byte(img,quality_factor):
    bytesIO=io.BytesIO()
    im = Image.fromarray(img)
    im.save(bytesIO,'JPEG',quality=quality_factor)
    jpeg_im = Image.open(bytesIO).convert('RGB')
    return bytesIO.getvalue(), np.array(jpeg_im, dtype='uint8')


def manipulate(org_img,manipulate_type, k=5, frac=1.0):
    """
    Manipulation Type
    {gaussBlur, median, blur, gaussNoise, histoEq, motionBlur, motionBlur2}
    """
    if manipulate_type == "blur":
        # 2-D Gaussian smoothing kernel: gaussian_filter(img, sigma=(n))
        #sigma = 1 + random.random()*2
                sigma = 2 * frac
                forged_img = cv2.GaussianBlur(org_img,(5,5),sigma)
                #print ("apply gaussian blur (sigma=%.2f)" % sigma)
                return np.array(forged_img, dtype='uint8')

    elif manipulate_type == "median":
        forged_img = cv2.medianBlur(org_img,k)
        return np.array(forged_img, dtype='uint8')

    elif manipulate_type == "noise":
        [row, col, c] = org_img.shape
        mean = 0
        sigma = 4 * frac
        gauss = np.random.normal(mean, sigma, (row, col, c))
        forged_img = org_img + gauss

        # delete 256+ and - values
        forged_img = delete_outrange(forged_img)
        return np.array(forged_img, dtype='uint8')

    elif manipulate_type == "resize":
        [row, col, c] = org_img.shape
        if frac == 1.0:
            fx = 1.5
            fy = 1.5
        else:
            fx = 1.2
            fy = 1.2
        forged_img = cv2.resize(org_img,None,fx=fx,fy=fy,interpolation=cv2.INTER_LINEAR)
        forged_img = forged_img[0:row,0:col,:]

    return np.array(forged_img, dtype='uint8')



'''

def manipulate_color_image(org_img,manipulate_type):
    if manipulate_type == "median_5":
        forged_img = cv2.medianBlur(org_img,5)
    
    elif manipulate_type == "resize_linear_12":
        forged_img = cv2.resize(org_img,None,fx=1.2,fy=1.2,interpolation=cv2.INTER_LINEAR)

    elif manipulate_type == "blur_5":
        forged_img = cv2.blur(org_img,(5,5))

    elif manipulate_type == "gaussNoise_2":
        [row, col,c] = org_img.shape
        mean = 0
        sigma = 2
        gauss = np.random.normal(mean,sigma,(row,col,c))
        forged_img = org_img + gauss

        #delete 256+ and - values
        forged_img = delete_outrange(forged_img)

    elif manipulate_type == "histoEq":
        img_yuv = cv2.cvtColor(org_img, cv2.COLOR_BGR2YUV)
        
        # equalize the histogram of the Y channel
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        
        # convert the YUV image back to RGB format
        forged_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return forged_img

def manipulate(org_img,manipulate_type):
    """
    Manipulation Type
    {gaussBlur, median, blur, gaussNoise, histoEq, motionBlur, motionBlur2}
    """
    if manipulate_type == "gaussBlur":
        # 2-D Gaussian smoothing kernel: gaussian_filter(img, sigma=(n))
        forged_img = ndimage.gaussian_filter(org_img, sigma=1.1)
        #forged_img = cv2.GaussianBlur(org_img,(5,5),sigmaX=1.1)

    elif manipulate_type == "resampling_105":
        b_size = org_img.shape[0]
        b_start = int(b_size*0.025)

        #resize
        resized_img = misc.imresize(org_img,1.05,interp='bilinear')
        #crop (center location)
        forged_img = resized_img[b_start:b_start+b_size,b_start:b_start+b_size]

    elif manipulate_type == "resampling_12":
        b_size = org_img.shape[0]
        b_start = int(b_size*0.1)

        #resize
        resized_img = misc.imresize(org_img,1.2,interp='bilinear')
        #crop (center location)
        forged_img = resized_img[b_start:b_start+b_size,b_start:b_start+b_size]

    elif manipulate_type == "sharpening_3":
        kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        forged_img = cv2.filter2D(org_img, -1, kernel)

    elif manipulate_type == "sharpening_5":
        kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        forged_img = cv2.filter2D(org_img, -1, kernel)

    elif manipulate_type == "median_3":
        forged_img = cv2.medianBlur(org_img,3)

    elif manipulate_type == "median_5":
        forged_img = cv2.medianBlur(org_img,5)

    elif manipulate_type == "blur_0.5":
        kernel = np.array([[0.0248,0.1079,0.0248],[0.1079,0.4685,0.1079],[0.0248,0.1079,0.0248]])
        forged_img = cv.filter2D(org_img, -1, kernel)

    elif manipulate_type == "blur_A":
        kernel = np.array([[0.1111,0.1111,0.1111],[0.1111,0.1111,0.1111],[0.1111,0.1111,0.1111]])
        forged_img = cv2.filter2D(org_img, -1, kernel)

    elif manipulate_type == "blur_3":
        kernel = np.array([[0.0113, 0.0838, 0.0113], [0.0838, 0.6193, 0.0838], [0.0113, 0.0838, 0.0113]]) #about sigma=0.4
        #kernel = np.array([[0.077847, 0.123317, 0.077847], [0.123317, 0.195346, 0.123317], [0.077847, 0.123317, 0.077847]]) # sigma = 1
        forged_img = cv2.filter2D(org_img, -1, kernel)
        #forged_img = cv2.blur(org_img,(3,3))

    elif manipulate_type == "blur_5":
        forged_img = cv2.blur(org_img,(5,5))

    elif manipulate_type == 'gaussNoise_0.5':
        row = org_img.shape[0]
        col = org_img.shape[1]
        mean = 0
        sigma = 0.5
        gauss = np.random.normal(mean,sigma,(row,col))
        gauss = gauss.reshape(row,col)
        forged_img = org_img + gauss
        forged_img = delete_outrange(forged_img)

    elif manipulate_type == 'gaussNoise_1':
        #print "org img shape", org_img.shape
        row = org_img.shape[0]
        col = org_img.shape[1]
        mean = 0
        sigma = 1
        gauss = np.random.normal(mean,sigma,(row,col))
        gauss = gauss.reshape(row,col)
        forged_img = org_img + gauss
        forged_img = delete_outrange(forged_img)

    elif manipulate_type == 'gaussNoise_1.5':
        #print "org img shape", org_img.shape
        row = org_img.shape[0]
        col = org_img.shape[1]
        mean = 0
        sigma = 1.5
        gauss = np.random.normal(mean,sigma,(row,col))
        gauss = gauss.reshape(row,col)
        forged_img = org_img + gauss
        forged_img = delete_outrange(forged_img)

    elif manipulate_type == 'gaussNoise_2':
        #print "org img shape", org_img.shape
        row = org_img.shape[0]
        col = org_img.shape[1]
        mean = 0
        sigma = 2
        gauss = np.random.normal(mean,sigma,(row,col))
        gauss = gauss.reshape(row,col)
        forged_img = org_img + gauss
        forged_img = delete_outrange(forged_img)

    elif manipulate_type == "histoEq":
        forged_img = cv2.equalizeHist(org_img)
        #forged_img = np.hstack((org_img,equ))

    elif manipulate_type == "motionBlur":
        kernel = 2*np.array([[0.1, 0, 0, 0, 0],[0, 0.1, 0, 0, 0],[0, 0, 0.1, 0, 0],[0, 0, 0, 0.1, 0],[0, 0, 0, 0, 0.1]])
        #kernel = dweight*np.array([[1,0,0],[0,1,0],[0,0,1]])
        kernel = kernel[:,:,None]
        forged_img = ndimage.convolve(org_img,kernel,mode='constant')
    
    elif manipulate_type == "motionBlur2":
        kernel = np.array([[0.1, 0, 0, 0, 0.1],[0, 0.1, 0, 0.1, 0],[0, 0, 0.1, 0, 0],[0, 0.1, 0, 0.1, 0],[0.1, 0, 0, 0, 0.1]])
        kernel = kernel[:,:,None]
        #kernel = dweight*np.array([[1,0,0],[0,1,0],[0,0,1]])
        forged_img = ndimage.convolve(org_img,kernel,mode='constant')

    elif manipulate_type == "resize":
        forged_img = misc.imresize(org_img,1.5,interp='bilinear')

    return forged_img

# Manipulate input blocks
# OUTPUT : [t_blocks,t_labels] = manipulate_blocks(blocks[0],labels,manipulation_type)
def manipulate_blocks(blocks, labels, manipulate_type, shuffle):
    assert len(blocks) == len(labels)

    # Info of blocks & labels
    batch_size = len(blocks)

    c = blocks.shape[3]
    block_size = blocks.shape[1]

    # Init empty ndarray
    t_blocks = np.zeros((batch_size*2,block_size,block_size,c))
    t_blocks = t_blocks.astype('uint8')

    t_labels = np.zeros((batch_size*2,2))

    # lables for forged blocks
    forged_labels = np.zeros((batch_size,2))
    forged_labels[:,1] = 1   

    # forged blocks ndarray
    forged_blocks = np.zeros((batch_size,block_size,block_size,c))
    forged_blocks = forged_blocks.astype('uint8')

    for i in range(batch_size): 
        org_img = blocks[i,:,:,:]

    # Manipulation Type
        forged_img = manipulate(org_img,manipulate_type)
        #print forged_img.shape
        #print forged_img[0,0]

        forged_img = forged_img.reshape([block_size,block_size,c])        
        # Allocate manipulated blocks
        forged_blocks[i] = forged_img

        # Test
        plt.imsave('img/org_'+str(i)+'.jpg',org_img)
        plt.imsave('img/forged_'+str(i)+'.jpg',blur_img)

    # Concatenate blocks & labels
    t_blocks = np.concatenate((blocks,forged_blocks),axis=0)
    t_labels = np.concatenate((labels,forged_labels),axis=0)

    # Shuffle two arrays in unison
    if shuffle == 1:
        [t_blocks, t_labels] = unison_shuffle(t_blocks,t_labels)

    return t_blocks,t_labels

def jpeg_compression(img,quality_factor):
    block_size = img.shape[0]
    img = img.reshape([block_size,block_size]) #gray image
    im = Image.fromarray(img)
    temp_path = "./temp.jpg"
    quality_factor = int(quality_factor)

    if quality_factor == -1:
        rq = random.randrange(1,12) # 1~11
        quality_factor = 45 + rq*5 # quality factor : 50~100
    elif quality_factor == -2:
        rq = random.randrange(0,51) # 0~50
        quality_factor = 50 + rq

    im.save(temp_path,'JPEG',quality=quality_factor)

    #read image
    read_img = misc.imread(temp_path)
    return read_img

def jpeg_compression_blocks(blocks, labels, quality_factor):
    assert len(blocks) == len(labels)

    n = len(blocks)
    c = blocks.shape[3]
    block_size = blocks.shape[1]

    #empty ndarray
    j_blocks = np.zeros((n,block_size,block_size,c))
    j_blocks = j_blocks.astype('uint8')
    for i in range(n):
        img = blocks[i,:,:,:]
        
        #jpeg compression
        jpg_img = jpeg_compression(img,quality_factor)
        jpg_img = jpg_img.reshape([block_size,block_size,c])
        j_blocks[i] = jpg_img

    j_labels = labels	

    return j_blocks,j_labels

'''