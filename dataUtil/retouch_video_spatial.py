
import argparse
import glob
from os import makedirs
from os.path import isdir, join, basename
from manipulate import manipulate
import skvideo.io as vio # pip install sk-video
import numpy as np
import cv2

def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out

def main():

    # parse the arguments
    parser = argparse.ArgumentParser(description='Retouch video.')
    parser.add_argument('--src_path', type=str, default='../27M', help='source path')
    parser.add_argument('--gt_path', type=str, default='../gt', help='ground truth path')
    parser.add_argument('--dst_path', type=str, default='../retouch_spatial', help='destination path')
    parser.add_argument('--intensity', type=str, default='extreme', help='strong or weak')
    args = parser.parse_args()

    # source directory validation check
    src_path    = args.src_path
    gt_path     = args.gt_path
    dst_path    = args.dst_path
    intensity   = args.intensity

    # method validation check
    bitrate = "27M"
    methods = ["blur", "median", "noise"]
    # methods = ["original"]

    # set destination directory name
    for method in methods:
        try:
            makedirs(join(dst_path, intensity, method))
            pass
        except FileExistsError:
            pass

    counter = 1
    fnames = glob.glob(join(src_path, "*.mp4"))
    gt_names = glob.glob(join(gt_path, "*.png"))
    print("%8s| file name" % "counter")

    # retouch video
    for fname in fnames:
        gt = None
        for gt_name in gt_names: 
            if basename(fname).split('.')[0] in gt_name: 
                gt = im2double(cv2.imread(gt_name, cv2.IMREAD_COLOR))
        # video read
        meta = vio.ffprobe(fname)
        vid = np.array(vio.vread(fname))
        vid_retouched = np.zeros(vid.shape)
        fn, w, h, c = vid.shape

        # parse bitrate from file name
        

        for method in methods:
            # get manipulated frame 
            for i in range(fn):
                retouched = manipulate(vid[i,:,:,:], method, intensity=intensity) # manipulate.py 참고
                original = vid[i,:,:,:]
                retouched_spatial = gt * retouched + (1 - gt) * original
                vid_retouched[i,:,:,:] = retouched_spatial

            vid_retouched = vid_retouched.astype(np.uint8)

            
            
            # load writer with parameter
            # "-vcodec = libx264"   : h.264 codec
            # "-r = 30"             : fps
            # "-g = 4"              : GOP size
            # "-bf = 0"             : number of b frame
            # "-b:v = bitrate"      : bitrate
            # "-pix_fmt = yuv420p"  : color space
            output_file = join(dst_path, intensity, method, basename(fname))
            write_option = {'-vcodec': 'libx264', '-r': '30', '-g': '4', '-bf': '0', '-b:v': bitrate, '-pix_fmt': 'yuv420p'}
            writer = vio.FFmpegWriter(filename=output_file, inputdict={'-r': '30'}, outputdict=write_option)
            for i in range(fn):
                writer.writeFrame(vid_retouched[i, :, :, :])
            # writer.writeFrame(vid_retouched)
            writer.close()

            # set output file name
            
            print("%8d: %s" % (counter , output_file))
            counter += 1

    print("Process end on directory \"%s\"" % src_path)

'''
    # for debug
    if False:
        print(basename(output_file))

        input_file = join(src_path + basename(output_file))
        input_meta = vio.ffprobe(input_file)
        for x in input_meta:
            print(x)
            for y in input_meta[x]:
                print (y, ':', input_meta[x][y])
        
        output_meta = vio.ffprobe(output_file)
        for x in output_meta:
            print(x)
            for y in output_meta[x]:
                print (y, ':', output_meta[x][y])
'''

    

    
if __name__=="__main__":
    main()