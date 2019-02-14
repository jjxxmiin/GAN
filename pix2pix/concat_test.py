import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('G:/dataset/maps/train/1.jpg',cv2.IMREAD_COLOR)


real = img[:,:600,:]
target = img[:,600:,:]

img_conv = np.concatenate((real,target),axis=2)

print(img_conv.shape)

cv2.imshow('ttt',img_conv[:,:,:2])
cv2.waitKey(0)
cv2.destroyAllWindows()
