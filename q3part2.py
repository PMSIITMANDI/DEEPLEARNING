#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 18:48:02 2019

@author: pushap
"""

import cv2
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import models
from keras.preprocessing import image
#import tensorflow as tf
import os
import shutil
from data_preprocessing1 import data_preprocessing




try:
    shutil.rmtree("lineDataset_weightFile")
except OSError as e:
    #print ("Error: %s - %s." % (e.filename, e.strerror))
    k=1
    
    
os.mkdir("lineDataset_weightFile")
currentDir = os.getcwd()

x_train,x_test,y_train,y_test = data_preprocessing("/home/pushap/DEEPLEARNINGLOCAL/ASSIGNMENT2/dataset/")
print(np.shape(x_train))
print(np.shape(y_train))
print(np.shape(x_test))
print(np.shape(y_test))
IMG_SIZE = (28, 28, 3)
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
x=model.add(Conv2D(64, kernel_size=(7,7), strides=1, activation='relu',input_shape=input_shape))
y=model.add(Conv2D(64, kernel_size=(3,3), strides=1, activation='relu',input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Flatten(data_format=None))
model.add(BatchNormalization())
model.add(Dense(2048, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(96,activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
filepath = currentDir + "/lineDataset_weightFile"+ "/lineset-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
model.fit(x=x_train,y=y_train, epochs=1, batch_size=100, callbacks= callbacks_list)

for layer in model.layers:
	# check for convolutional layer
	if 'conv' not in layer.name:
		continue
	# get filter weights
	filters, biases = layer.get_weights()
	print(layer.name, filters.shape)

filters, biases = model.layers[1].get_weights()
# normalize filter values to 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
# plot first few filters
n_filters, ix = 6, 1
for i in range(n_filters):
	# get the filter
	f = filters[:, :, :, i]
	# plot each channel separately
	for j in range(3):
		# specify subplot and turn of axis
		ax = pyplot.subplot(n_filters, 3, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		pyplot.imshow(f[:, :, j], cmap='gray')
		ix += 1
# show the figure
pyplot.show()

img_tensor = image.img_to_array(x_test[0])
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.
layer_outputs = [layer.output for layer in model.layers[:12]] 
# Extracts the outputs of the top 12 layers

activation_model = models.Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
activations = activation_model.predict(img_tensor) 
# Returns a list of five Numpy arrays: one array per layer activation
first_layer_activation = activations[0]
print(first_layer_activation.shape)


layer_names = []
for layer in model.layers[:3]:
    layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
    
images_per_row = 16
for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
    n_features = layer_activation.shape[-1] # Number of features in the feature map
    size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
    n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols): # Tiles each filter into a big horizontal grid
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
            channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, # Displays the grid
                         row * size : (row + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')

#flower_output = model.output[:, 0]
#last_conv_layer = model.get_layer('conv2d_2')
#
#grads = K.gradients(flower_output, last_conv_layer.output)[0]
#pooled_grads = K.mean(grads, axis=(0, 1, 2))
#iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
#pooled_grads_value, conv_layer_output_value = iterate([x])
#
#
#
#for i in range(64):
#	conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
#
#heatmap = np.mean(conv_layer_output_value, axis=-1)
#heatmap = np.maximum(heatmap, 0)
#heatmap /= np.max(heatmap)
#plt.savefig(heatmap)
#
##Using cv2 to superimpose the heatmap on original image to clearly illustrate activated portion of image
#img = cv2.imread(x_test[0])
#heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
#heatmap = np.uint8(255 * heatmap)
#heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#superimposed_img = heatmap * 0.4 + img
#cv2.imwrite('image_name.jpg', superimposed_img)