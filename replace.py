
f = open("test_weak.txt", 'r')
fnames = f.read().splitlines()
f_out = open("test_weak_.txt", 'w')
base = r"\\DESKTOP-F3SBHR2"

for fname in fnames:
	r = [base] + fname.split("\\")[-5:]
	r_ = "\\".join(r)
	f_out.write(r_+"\n")