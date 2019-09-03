import tensorflow as tf
import numpy as np
import imageio
import random
from os import listdir, makedirs, cpu_count

import argparse
from os.path import join, isfile, isdir, splitext, basename
from joblib import Parallel, delayed
import glob
import re


# Debug function
def print_list(l):
    for c in l:
        print(c)
    print()


def print_dict(d):
    for c in d:
        print(c, d[c])
    print()


# bytes conversion functions
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _int64list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def file_name_br(fname):
    video_name = splitext(basename(fname))[0]
    br = int(video_name.split('_')[4])
    return video_name, br


def file_method_label(fname):
    # classify the label vector dimension by number of class
    method = fname.split("\\")[-2]
    if method == "original":
        label = [1,0,0,0,0]
    elif method == "blur":
        label = [0,1,0,0,0]
    elif method == "median":
        label = [0,0,1,0,0]
    elif method == "noise":
        label = [0,0,0,1,0]
    elif method == "resize":
        label = [0,0,0,0,1]
    else:
        raise(BaseException("no such method \"{}\"").format(method))


    return method, np.array(label, dtype=np.uint8)



def video_read(bundle, output_dir, stack_per_video, stack_size):
    # input video placeholder
    video_bundle = {fname: np.zeros([stack_per_video, stack_size, 256, 256, 3], dtype=np.uint8) for fname in bundle}
    
    # Extract video name and bitrate
    for video_sample in bundle:
        video_name, br = file_name_br(video_sample)
        break

    # read and store decoded video
    for fname, buffers in video_bundle.items():
        with imageio.get_reader(fname, 'ffmpeg') as video:
            video_len = video.count_frames()
            frame_nums = random.sample(range(0, video_len - stack_size), stack_per_video)

            for frame_idx, frame_num in enumerate(frame_nums):
                buffers[frame_idx] = (np.array([video.get_data(frame_num + idx) for idx in range(stack_size)]))


    # Write the record
    for frame_num in range(stack_per_video):
        tfrecord_name = join(output_dir, "{}.tfrecord").format(br, video_name + str(frame_num))
        print(tfrecord_name)
        with tf.python_io.TFRecordWriter(tfrecord_name) as writer:
            for fname, buffers in video_bundle.items():
                buffer = buffers[frame_num]
                method, label = file_method_label(fname)

                features = {
                    # mendatory informations
                    'frames' : _bytes_feature(buffer.tobytes()),
                    'label' : _bytes_feature(label.tobytes()),

                    # additional informations
                    # 'br' : _int64_feature(br)
                }

                example = tf.train.Example(features = tf.train.Features(feature=features))
                writer.write(example.SerializeToString())


def build_tfrecord():

    parser = argparse.ArgumentParser(description='make tfrecord.')
    parser.add_argument('--src_path', type=str, default='./trainS_output', help='source path')
    parser.add_argument('--dst_path', type=str, default='./retouch_tfrecord_trainS', help='dst path')
    parser.add_argument('--stack_per_video', type=int, default=15, help='number of stack in one video')
    parser.add_argument('--stack_size', type=int, default=1, help='size of stack')
    args = parser.parse_args()


    stack_per_video = args.stack_per_video
    stack_size = args.stack_size

    print(vars(args))

    # default setup
    src_path = args.src_path
    dst_path = args.dst_path

    methods = ["blur", "median", "noise", "resize"]
    brs = [500,600,700,800]


    # create save directory
    output_dir = join(dst_path, "{}br")
    try:
        for br in brs:
            dirname = output_dir.format(str(br))
            makedirs(dirname)
    except Exception as e:
        pass

    original_fnames = glob.glob(join(src_path, "original", "*.mp4"))
    fnames = {original_fname: glob.glob(join(src_path, "*", basename(original_fname))) for original_fname in original_fnames}


    # Record
    Parallel(n_jobs=cpu_count())(delayed(video_read)\
        (fname, output_dir, stack_per_video, stack_size) \
            for _, fname in fnames.items())

    # for _, fname in fnames.items():
    #     video_read(fname, output_dir, stack_per_video, stack_size)

    print("End processing on directory\"{}\"".format(src_path))

if __name__ == '__main__' :
    build_tfrecord()
