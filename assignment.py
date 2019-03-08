import numpy as np
import cv2

img_row=28
img_col=28
blue =(255,0,0)
red=(0,0,255)
thin = 1
thick =3
short =7
long =15

length = long
width = thin
theta=125
color =blue

tx=14
ty=18


#############################################
testimage = np.zeros((img_row,img_col,3), np.uint8)

if (length==short):
    s_i=10
    s_j=13
    
    e_i=17
    e_j=13
else:
    s_i=10
    s_j=13
    
    e_i=25
    e_j=13
    


testimage = cv2.line(testimage,(s_i,s_j),(e_i,e_j),color,width)
rotationMatrix = cv2.getRotationMatrix2D((img_row/2,img_col/2),theta,1)
translationMatrix = np.float32([[1, 0, 100], [0, 1, 50]]) 
testimage = cv2.warpAffine(testimage,rotationMatrix,(img_row,img_col))
cv2.imshow('testimage',testimage)
cv2.waitKey(0)