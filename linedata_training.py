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
import shutil





try:
    shutil.rmtree("lineDataset_weightFile")
except OSError as e:
    #print ("Error: %s - %s." % (e.filename, e.strerror))
    k=1
    
    
os.mkdir("lineDataset_weightFile")
currentDir = os.getcwd()

x_train,x_test,y_train,y_test = data_preprocessing("./dataset/")
print(np.shape(x_train))
print(np.shape(y_train))
print(np.shape(x_test))
print(np.shape(y_test))

#image_index = 4000 # You may select anything up to 60,000
#print(y_train[image_index]) # The label is 8
#cv2.imshow('image',x_train[4000])
#cv2.waitKey(0)
# print(np.shape(x_train))
# print(np.shape(y_train))
#
x_train = x_train.reshape(x_train.shape[0], 28, 28, 3)#converts 2D to 3D in batch.
x_test = x_test.reshape(x_test.shape[0], 28, 28, 3)#converts 2D to 3D in batch
input_shape = (28, 28, 3)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# # Normalizing the RGB codes by dividing it to the max RGB value.
x_train = x_train/255
# x_test = x_test/255
# print('x_train shape:', x_train.shape)
# print('Number of images in x_train', x_train.shape[0])
# print('Number of images in x_test', x_test.shape[0])
#
y_train = np_utils.to_categorical(y_train)
#
model = Sequential()
model.add(Conv2D(64, kernel_size=(7,7), strides=1, activation=tf.nn.relu,input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(3,3), strides=1, activation=tf.nn.relu,input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Flatten(data_format=None))
model.add(BatchNormalization())
model.add(Dense(2048, activation=tf.nn.relu))
model.add(BatchNormalization())
model.add(Dense(96,activation=tf.nn.softmax))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
filepath = currentDir + "/lineDataset_weightFile"+ "/lineset-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
model.fit(x=x_train,y=y_train, epochs=3, batch_size=100, callbacks= callbacks_list)
