import tensorflow as tf
import numpy as np
import PIL
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.layers.normalization import BatchNormalization

import os


currentDir = os.getcwd()
path = currentDir + "/MNIST/*"


weightFiles = os.listdir("./mnist_weightFile/")
weightFiles.sort()
weightFilepath = currentDir+"/mnist_weightFile/"+weightFiles[3]



(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path)

image_index1 = 1
img = PIL.Image.fromarray(x_test[image_index1])
img.show()
print("X-train shape: "+ str(np.shape(x_train)))
print("Y-train shape: "+ str(np.shape(y_train)))
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)#converts 2D to 3D in batch
input_shape = (28, 28, 1)
x_test = x_test/255

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), strides=1, activation=tf.nn.relu,input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Flatten(data_format=None))
model.add(Dense(1024, activation=tf.nn.relu))
# model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))


model.load_weights(weightFilepath)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

x_test[image_index1]=x_test[image_index1]*255

pred = model.predict(x_test[image_index1].reshape(1, 28, 28, 1))

print("****************************************************************************")
print("Predicted output: "+ str(pred.argmax()))
