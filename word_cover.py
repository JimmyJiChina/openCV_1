import numpy as ny
import cv2
import matplotlib.pyplot as mplot
color_sel = 0
filename = 'word_capture.jpg'
fid = cv2.imread(filename,color_sel)
#fid_re = cv2.resize(fid,(1024,1024))
cv2.imshow('image',fid)
#cv2.waitKey(0)
f = ny.fft.fft2(fid)
m,n = f.shape
f[m/2-2:m/2+1:2,n/2-400:n/2+400]=0
fid1 = ny.fft.ifft2(f)
fid1_r = ny.uint8(fid1.real)
cv2.imshow('image1',fid1_r)
cv2.waitKey(0)
cv2.destroyAllWindows()
sb_file = 'sb.jpg'
fid_sb_orignal = cv2.imread(sb_file,color_sel)
fid_sb_orignal = cv2.resize(fid_sb_orignal,(40,40))
fid_sb_2l = ny.reshape(fid_sb_orignal,(2,800))
#f[m/2,n/2-100:n/2+100]=fid_sb_1l
f[m/2-2:m/2+1:2,n/2-400:n/2+400]=fid_sb_2l
#f_replace=fid_sb_2l
fid_replace = ny.fft.ifft2(f)
fid_sb = ny.uint8(fid_replace.real)
cv2.imshow('sb',fid_sb)
cv2.waitKey(0)

fid_new = fid_sb

f_new = ny.fft.fft2(fid_new)
f_sb1_re = f_new[m/2-2:m/2+1:2,n/2-400:n/2+400]
fid_sb_show = f_sb1_re
fid_sb_show = ny.reshape(fid_sb_show,(40,40))
fid_sb_show = ny.uint8(fid_sb_show.real)
#fid_sb = cv2.resize(fid_sb,(200,200))
cv2.imshow('sb2',fid_sb_show)
cv2.waitKey(0)
cv2.destroyAllWindows()
