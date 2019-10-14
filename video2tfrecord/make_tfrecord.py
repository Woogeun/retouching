import tensorflow as tf
import numpy as np
import imageio
import random
from os import listdir, makedirs, cpu_count

import argparse
from os.path import join, isfile, isdir, splitext, basename
from joblib import Parallel, delayed
from glob import glob
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


def name_split(fname):
    splited = fname.split('\\')
    br = splited[-2]
    name = splited[-1]
    return br, name


def file_name_br(fname):
    video_name = splitext(basename(fname))[0]
    br = fname.split('\\')[-2]
    return video_name, br


def extract_information(bundle, stack_per_video, stack_size):
    for sample in bundle:
        video_name, br = file_name_br(sample)
        with imageio.get_reader(sample, 'ffmpeg') as video:
            video_len = video.count_frames()
            frame_nums = random.sample(range(0, video_len - stack_size), stack_per_video)
        return video_name, br, frame_nums


def file_method_label(fname):
    # classify the label vector dimension by number of class
    method = fname.split("\\")[-3]
    if method == "original":
        label = [1,0,0,0]
    elif method == "blur":
        label = [0,1,0,0]
    elif method == "median":
        label = [0,0,1,0]
    elif method == "noise":
        label = [0,0,0,1]
    else:
        raise(BaseException("no such method \"{}\"").format(method))

    # if method == "original":
    #     label = [1,0]
    # else:
    #     label = [0,1]

    return method, np.array(label, dtype=np.uint8)



def video_read(bundle, output_dir, stack_per_video, stack_size):
    METHOD, _ = file_method_label(bundle[1])

    # input video placeholder
    video_bundle = {fname: np.zeros([stack_per_video, stack_size, 256, 256, 3], dtype=np.uint8) for fname in bundle}
    

    # Extract video name and bitrate
    video_name, br, frame_nums = extract_information(bundle, stack_per_video,stack_size)
    

    # read and store decoded video
    for fname, buffers in video_bundle.items():
        with imageio.get_reader(fname, 'ffmpeg') as video:
            for frame_idx, frame_num in enumerate(frame_nums):
                buffers[frame_idx] = (np.array([video.get_data(frame_num + idx) for idx in range(stack_size)]))


    # Write the record
    for frame_num in range(stack_per_video):
        tfrecord_name = join(output_dir, br, "{}_{}_{}.tfrecord").format(video_name, METHOD, str(frame_num))
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
    parser.add_argument('--src_path', type=str, default='./retouch_strong', help='source path')
    parser.add_argument('--dst_path', type=str, default='./tfrecord_retouch_strong', help='dst path')
    parser.add_argument('--stack_per_video', type=int, default=16, help='number of stack in one video')
    parser.add_argument('--stack_size', type=int, default=1, help='size of stack')
    parser.add_argument('--attack', type=str, default='multi', help='blur, median, noise or multi')
    args = parser.parse_args()


    stack_per_video = args.stack_per_video
    stack_size = args.stack_size
    attack = args.attack

    print_dict(vars(args))

    # default setup
    src_path = args.src_path # E:\paired_minibatch\retouch_strong
    dst_path = args.dst_path # E:\paired_minibatch\tfrecord_retouch_strong
    dst_path = join(dst_path, attack) # E:\paired_minibatch\tfrecord_retouch_strong\some_attack

    if attack != "multi":
        methods = [attack]
    else:
        methods = ["blur", "median", "noise"]
    bitrates = ["500k","600k","700k","800k"]


    # create save directory
    try:
        for bitrate in bitrates:
            makedirs(join(dst_path, bitrate))
    except Exception as e:
        pass


    original_fnames = glob(join(src_path, "original", "*", "*.mp4"))

    fnames = []
    for original_fname in original_fnames:
        br, name = name_split(original_fname)
        for method in methods:
            retouched = glob(join(src_path, method, br, name))
            assert(len(retouched) == 1)
            fnames.append((original_fname, retouched[0]))

    print_list(fnames)

    # Record
    # Parallel(n_jobs=cpu_count())(delayed(video_read)\
    #     (bundle, dst_path, stack_per_video, stack_size) \
    #         for bundle in fnames)

    # for bundle in fnames:
    #     video_read(bundle, dst_path, stack_per_video, stack_size)

    print("End processing on directory\"{}\"".format(src_path))




if __name__ == '__main__' :
    build_tfrecord()
