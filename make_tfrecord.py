import tensorflow as tf
import numpy as np
import imageio
import random
from os import listdir, makedirs


from os.path import join, isfile, isdir
from joblib import Parallel, delayed
import glob
import codecs
import re


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _int64list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))





def video_read_train(file, batch_size, stack_size):
    split_ = file.split('\\')

    frames_arr = np.zeros([batch_size, stack_size, 256, 256, 3], dtype=np.uint8)

    video = imageio.get_reader(file, 'ffmpeg')

    video_len = video.get_length()
    if video_len - stack_size  < batch_size :
        return
    frame_nums = random.sample(range(0, video_len - stack_size), batch_size)

    for i in range(batch_size):
        frame_num = frame_nums[i]
        frames_arr[i] = np.array([video.get_data(frame_num + i) for i in range(4)])
    video.close()
    type = split_[-3]
    fname = split_[-1].split('.')[0]
    phase = split_[-4]
    method = split_[-5].split('/')[-1][-2:]
    fps = int(split_[-2][0:2])
    if not isdir('G:/tfrecord/{}/{}/{}/{}fps/'.format(method,phase, type,fps)):
        makedirs('G:/tfrecord/{}/{}/{}/{}fps/'.format(method,phase, type,fps), exist_ok=True)

    tfrname= 'G:/tfrecord/{}/{}/{}/{}fps/{}.tfrecord'.format(method,phase, type,fps, fname)
    if type =='Original':
        label = np.array([1,0], np.int32)
    else:
        label = np.array([0,1], np.int32)

    with tf.python_io.TFRecordWriter(tfrname) as writer:
        for i in range(batch_size):

            frame = frames_arr[i]

            features = {
                'frames' : _bytes_feature(frame.tobytes()),
                'label' : _bytes_feature(label.tobytes()),
                'fps' : _int64_feature(fps)
            }
            example = tf.train.Example(features = tf.train.Features(feature=features))
            writer.write(example.SerializeToString())



def video_read_test(file, batch_size):
    name = file.split('\\')
    frames_arr = np.zeros([7, 4, 256, 256, 3], dtype=np.uint8)

    video = imageio.get_reader(file, 'ffmpeg')

    video_len = video.get_length()
    frame_nums = random.sample(range(0, video_len - 4), int(batch_size / 2))

    for i in range(7):
        frame_num = frame_nums[i]

        frames_arr[i] = np.array([video.get_data(frame_num + i) for i in range(4)])
    video.close()


    fname = name[-1].split('mp4')[0][:-1]
    phase = name[0].split('/')[3]

    label = name[1]
    if label =='Original':
        label = np.array([1, 0], dtype=np.int32)
    else:
        label = np.array([0,1], dtype= np.int32)
    fps = int(name[2][0:2])
    tfrname= 'D:/tfrecord/NN/{}/{}_{}_{}.tfrecord'.format(phase, fname, name[1], fps)

    with tf.python_io.TFRecordWriter(tfrname) as writer:
        for i in range(7):

            frame = frames_arr[i]

            features = {
                'frames' : _bytes_feature(frame.tobytes()),
                'label' : _bytes_feature(label.tobytes()),
                'fps' : _int64_feature(fps)
            }
            example = tf.train.Example(features = tf.train.Features(feature=features))
            writer.write(example.SerializeToString())



def input_files(phase, method):
    s_dir = './data/crop_{}/{}/{}/{}fps'.format(method, phase)
    s_files = ['{}/{}'.format(s_dir, f) for f in listdir(s_dir) if isfile(join(s_dir, f))]
    return s_files


def build_tfrecord():

    method = "*"
    # method = "blur"
    # method = "median"
    # method = "noise"
    
    train_origianl_path = join("./train", "*.mp4")
    train_tampered_path = join("./train_output0", method, "*.mp4")

    test_origianl_path = join("./test", "*.mp4")
    test_tampered_path = join("./test_output0", method, "*.mp4")

    batch_size = 32
    stack_size = 1

    files = glob.glob(train_origianl_path) + glob.glob(train_tampered_path)
    # Parallel(n_jobs=8)(delayed(video_read_train)(file, batch_size, stack_size) for file in files)


    # files = glob.glob('./data/crop_{}/tra*/*/*/*.mp4'.format('MI'))
    # Parallel(n_jobs=8)(delayed(video_read_train)(file, 15) for file in files)

    # files = glob.glob('./data/crop_{}/te*/*/*/*.mp4'.format('MI'))
    # Parallel(n_jobs=8)(delayed(video_read_train)(file, 7) for file in files)


if __name__ == '__main__' :
    build_tfrecord()
