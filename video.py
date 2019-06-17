from manipulate import manipulate
import skvideo.io as vio # pip install sk-video
import numpy as np

meta = vio.ffprobe('mclv_S_0_1_500_0_.mp4')
vid = np.array(vio.vread('mclv_S_0_1_500_0_.mp4'))
fn, w, h, c = vid.shape


for i in range(fn):
    vid[i,:,:,:] = manipulate(vid[i,:,:,:], 'gaussNoise_2') # manipulate.py 참고




out = vid.astype(np.uint8)
writer = vio.FFmpegWriter(filename="out.mp4", outputdict={}) # outputdict에 bitrate, gop 등 추가
for i in range(fn):
        writer.writeFrame(out[i, :, :, :])
writer.close() 