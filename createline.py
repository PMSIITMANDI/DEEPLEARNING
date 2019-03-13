from PIL import Image
import numpy as np


d=0
for i in range(0,2):
	for j in range(0,2):
		rgbArray = np.zeros((28,28,3), 'uint8')
		rgbArray[..., 0] = 0
		rgbArray[..., 1] = 0
		rgbArray[..., 2] = 0
		rgbArray[i,j:j+6,0]=255
		img = Image.fromarray(rgbArray)
		img.save('myimg%d.jpeg'%d)
		d=d+1
		x=np.sum(img)
		print(x)