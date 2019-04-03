from data_preprocessing1 import data_preprocessing
import tensorflow as tf
import numpy as np
import cv2
import PIL
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import os

currentDir = os.getcwd()

weightFiles = os.listdir("./lineDataset_weightFile/")
weightFiles.sort()
weightFilepath = currentDir+"/lineDataset_weightFile/"+weightFiles[4]


x_train,x_test,y_train,y_test = data_preprocessing("./dataset/")

input_shape = (28, 28, 3)

image_index1= 13054


# y_test = np_utils.to_categorical(y_test)

x_test = x_test/255

model = Sequential()
model.add(Conv2D(64, kernel_size=(7,7), strides=1, activation=tf.nn.relu,input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(3,3), strides=1, activation=tf.nn.relu,input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Flatten(data_format=None))
# model.add(BatchNormalization())
# model.add(Dense(512, activation=tf.nn.relu))
# model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(2048, activation=tf.nn.relu))
# model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(96,activation=tf.nn.softmax))
model.load_weights(weightFilepath)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# z=model.evaluate(x_test,y_test,batch_size=100,verbose=1)
# print(z)
# x_test[image_index1]=x_test[image_index1]*255
# # # cv2.imshow('image',x_test[image_index1])
# # # cv2.waitKey(0)
pred = model.predict(x_test[image_index1].reshape(1, 28, 28, 3))
print("***********************************************************************************************")
print("Predicted Output: "+ str(pred.argmax()))
print(y_test[image_index1])
# # # cv2.imshow('image',x_test[image_index1])
# # # cv2.waitKey(0)
