import tensorflow as tf
import numpy as np
import PIL
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import regularizers
import os


currentDir = os.getcwd()
path = currentDir + "/MNIST/*"
no_of_epochs = 4

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path)
image_index = 4000 # You may select anything up to 60,000
print(y_train[image_index]) # The label is 8
x_train[image_index]=x_train[image_index]*255
img = PIL.Image.fromarray(x_train[image_index])
# img.show()
print(np.shape(x_train))
print(np.shape(y_train))
#
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)#converts 2D to 3D in batch.
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)#converts 2D to 3D in batch
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# # Normalizing the RGB codes by dividing it to the max RGB value.
x_train = x_train/255
x_test = x_test/255
# print('x_train shape:', x_train.shape)
# print('Number of images in x_train', x_train.shape[0])
# print('Number of images in x_test', x_test.shape[0])
#
y_train = np_utils.to_categorical(y_train)
#
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), strides=1, activation=tf.nn.relu,input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Flatten(data_format=None))
model.add(Dense(1024, activation=tf.nn.relu))
# model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
filepath = currentDir+"weightFile"+"/mnist-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
model.fit(x=x_train,y=y_train, epochs=no_of_epochs, batch_size=100, callbacks= callbacks_list)

