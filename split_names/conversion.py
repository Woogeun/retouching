from glob import glob
import random





def write(phase, fnames):
	with open(phase + ".txt", 'w') as f:
		for fname in fnames:
			f.write(fname + "\n")

def step(method, intensity):

	fnames = glob("E:\\paired_minibatch\\tfrecord_retouch_{}\\{}\\*\\*.tfrecord".format(intensity, method))
	random.shuffle(fnames)

	validFnames = fnames[:int(len(fnames) * 0.1)]
	testFnames = fnames[int(len(fnames) * 0.1): int(len(fnames) * 0.2)]
	trainFnames = fnames[int(len(fnames) * 0.2):]

	

	write("./{}/train_{}".format(method, intensity), trainFnames)
	write("./{}/test_{}".format(method, intensity), testFnames)
	write("./{}/valid_{}".format(method, intensity), validFnames)


for method in ["blur", "median", "noise", "multi"]:
	for intensity in ["strong", "weak"]:
		step(method, intensity)