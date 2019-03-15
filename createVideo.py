##import numpy as np
#import matplotlib.pyplot as plt
##import matplotlib.image as mpimg
##import 
##
##def rgb2gray(rgb):
##    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
##
##img = mpimg.imread('./short_thin_blue_0/0_0_0_1_1.jpg')     
##gray = rgb2gray(img)    
###plt.imshow(gray, cmap = plt.get_cmap('gray'))
###plt.show()
##
###I = cv2.imread("./short_thin_blue_0/0_0_0_1_1.jpg")
##plt.plot(img)
##plt.show()
#
#
#
import cv2
#import numpy as np
#
#img =cv2.imread("./short_thin_blue_0/0_0_0_1_1.jpg")
#img_array = np.asarray(img)
#plt.plot(img_array)
#plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
image1 = mpimg.imread("./dataset/short_thin_blue_0/0_0_0_1_1.jpg")
image2 = mpimg.imread("./dataset/short_thin_blue_0/0_0_0_1_100.jpg")
image3 = mpimg.imread("./dataset/short_thin_blue_0/0_0_0_1_200.jpg")
image4 = mpimg.imread("./dataset/short_thin_blue_0/0_0_0_1_300.jpg")
image5 = mpimg.imread("./dataset/short_thin_blue_0/0_0_0_1_400.jpg")
image6 = mpimg.imread("./dataset/short_thin_blue_0/0_0_0_1_500.jpg")
image7 = mpimg.imread("./dataset/short_thin_blue_0/0_0_0_1_600.jpg")
image8 = mpimg.imread("./dataset/short_thin_blue_0/0_0_0_1_700.jpg")
image9 = mpimg.imread("./dataset/short_thin_blue_0/0_0_0_1_800.jpg")


#plt.subplot(331)
#plt.imshow(image1)
#
#plt.subplot(332)
#plt.imshow(image2)
#
#plt.subplot(333)
#plt.imshow(image3)
#plt.subplot(334)
#plt.imshow(image4)
#plt.subplot(335)
#plt.imshow(image5)
#plt.subplot(336)
#plt.imshow(image6)
#
#plt.subplot(337)
#plt.imshow(image7)
#
#plt.subplot(338)
#plt.imshow(image8)


x= np.concatenate((image1,image2,image3),axis=1)
x1= np.concatenate((image4,image5,image6),axis=1)
x2= np.concatenate((image4,image5,image6),axis=1)

x3 = np.concatenate((x,x1,x2),axis=0)


#cv2.imshow("imshow",x)
#cv2.waitKey(0)

#x1= np.hstack(image1,image2,image3)
#x2=np.hstack(image4,image5,image6)
#x3 =np.hstack(image7,image8,image9)
#f=np.vstack(x1,x2,x3)

cv2.imshow("imshow",x3)
cv2.waitKey(0)

#plt.subplot(339)
#plt.imshow(image9)