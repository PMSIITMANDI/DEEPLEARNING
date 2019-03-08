import numpy as np
import cv2
import os

img_row=28
img_col=28
blue =(255,0,0)
red=(0,0,255)
thin = 1
thick =3
short =7
long =15
width = thin
color =blue




#############################################


os.mkdir("dataset")
testimage = np.zeros((img_row,img_col,3), np.uint8)
d=0
for i in range(6,7):
	for j in range(6,8):
		for theta in range(0,180,15):
			testimage = cv2.line(testimage,(j,i),(j+7,i),color,width)
			rotationMatrix = cv2.getRotationMatrix2D((i,j),theta,1)
			translationMatrix = np.float32([[1, 0, 100], [0, 1, 50]])
			testimage = cv2.warpAffine(testimage,rotationMatrix,(img_row,img_col))
			cv2.imwrite("./dataset/face-%d.jpg"%d,testimage)
			d=d+1
			testimage = np.zeros((img_row,img_col,3), np.uint8)
