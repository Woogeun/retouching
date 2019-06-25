import tensorflow as tf
import numpy as np
import imageio
import random
from os import listdir, makedirs


from os.path import join, isfile, isdir, splitext
from joblib import Parallel, delayed
import glob
import codecs
import re

counter = 1
num_of_class = -1


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _int64list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



def file_info(fname):
    fname_split = fname.split('\\')
    video_name = splitext(fname_split[-1])[0]
    
    fps = int(video_name.split('_')[4])

    label_name = fname_split[-2]

    if label_name not in ["blur", "median", "noise"]:
        label_name = "original"

    if num_of_class == 4:
        if label_name == "original":
            label = [1,0,0,0]
        elif label_name == "blur":
            label = [0,1,0,0]
        elif label_name == "median":
            label = [0,0,1,0]
        elif label_name == "noise":
            label = [0,0,0,1]
    elif num_of_class ==2:
        if label_name == "original":
            label = [1,0]
        else:
            label = [0,1]
    else:
        raise(BaseException("wrong num_of_class variable: %d" % num_of_class))

    return video_name, fps, label_name, np.array(label)



def video_read(fname, stack_per_video, stack_size):

    # input video placeholder
    frames_arr = np.zeros([stack_per_video, stack_size, 256, 256, 3], dtype=np.uint8)

    # read and store decoded video
    with imageio.get_reader(fname, 'ffmpeg') as video:
        video_len = video.get_length()

        # too small size video to generate batch size data
        if video_len - stack_size  < stack_per_video: 
            return

        # select frame samples for record ("stack_per_video" frames)
        frame_nums = random.sample(range(0, video_len - stack_size), stack_per_video)
        for i in range(stack_per_video):
            frame_num = frame_nums[i]
            frames_arr[i] = np.array([video.get_data(frame_num + i) for i in range(stack_size)])
    
    # extract file name and label vector
    video_name, fps, label_name, label = file_info(fname)

    # create save directory
    dirname = join("./retouch_tfrecord", label_name, "{}fps").format(fps)
    if not isdir(dirname):
        makedirs(dirname)

    # set tfrecord file name
    tfrecord_name = join(dirname, "{}.tfrecord").format(video_name)
    
    # save tfrecord file
    with tf.python_io.TFRecordWriter(tfrecord_name) as writer:
        for frame in frames_arr:
            features = {
                # mendatory informations
                'frames' : _bytes_feature(frame.tobytes()),
                'label' : _bytes_feature(label.tobytes()),

                # additional informations
                'fps' : _int64_feature(fps)
            }

            example = tf.train.Example(features = tf.train.Features(feature=features))
            writer.write(example.SerializeToString())
           

    global counter
    print("%8d: %s" % (counter , tfrecord_name))
    counter += 1
    

    # split_ = fname.split('\\')
    # type = split_[-3]
    # fname = split_[-1].split('.')[0]
    # phase = split_[-4]
    # method = split_[-5].split('/')[-1][-2:]
    # fps = int(split_[-2][0:2])
    # if not isdir('G:/tfrecord/{}/{}/{}/{}fps/'.format(method,phase, type,fps)):
    #     makedirs('G:/tfrecord/{}/{}/{}/{}fps/'.format(method,phase, type,fps), exist_ok=True)

    # tfrname= 'G:/tfrecord/{}/{}/{}/{}fps/{}.tfrecord'.format(method,phase, type,fps, fname)
    # if type =='Original':
    #     label = np.array([1,0], np.int32)
    # else:
    #     label = np.array([0,1], np.int32)

    


def input_files(phase, method):
    s_dir = './data/crop_{}/{}/{}/{}fps'.format(method, phase)
    s_files = ['{}/{}'.format(s_dir, f) for f in listdir(s_dir) if isfile(join(s_dir, f))]
    return s_files


def build_tfrecord():
    method = "*"
    # method = "blur"
    # method = "median"
    # method = "noise"

    global num_of_class
    num_of_class = 4 if method == "*" else 2
    
    train_origianl_path = join("./train", "*.mp4")
    train_tampered_path = join("./train_output0", method, "*.mp4")

    test_origianl_path = join("./test", "*.mp4")
    test_tampered_path = join("./test_output0", method, "*.mp4")

    stack_per_video = 15
    stack_size = 3

    fnames = glob.glob(train_origianl_path) + glob.glob(train_tampered_path)

    print("%8s| file name" % "counter")
    # for fname in fnames:
    #     video_read(fname, stack_per_video, stack_size)
    Parallel(n_jobs=4)(delayed(video_read)(fname, stack_per_video, stack_size) for fname in fnames)



if __name__ == '__main__' :
    build_tfrecord()
