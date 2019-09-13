
import skvideo.io as vio
import numpy as np
from manipulate import manipulate

vOriginal = np.array(vio.vread("../train_original/mclv_S_0_1_500_0_.mp4"))
vRetouched = np.zeros(vOriginal.shape)

fn, w, h, c = vOriginal.shape

start_fn = int(fn / 3)
end_fn = int(fn*2 / 3)

attack = "noise"

for idx in range(fn):
	if idx >= start_fn and idx <= end_fn:
		vRetouched[idx] = manipulate(vOriginal[idx], attack, intensity="strong")
	else:
		vRetouched[idx] = vOriginal[idx]


fps = '30'
write_option = {'-vcodec': 'libx264', '-r': fps, '-g': '4', '-bf': '0', '-pix_fmt': 'yuv420p'}
writer = vio.FFmpegWriter(filename="../mclv_S_0_1_500_0_{}.mp4".format(attack), inputdict={'-r': fps}, outputdict=write_option)
for frame in vRetouched:
	writer.writeFrame(frame)
writer.close()

