#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 23:56:37 2019

@author: pushap
"""
import sys
import glob
import os
import numpy as np
#import keras
#from keras.models import Sequential
#from keras.layers import Dense,Conv2D,Dropout,Flatten,MaxPooling2D
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers import Input
#import cv2
#from keras.callbacks import Modelcheckpoint
import cv2
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D,GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import models
from keras.preprocessing import image
import os
import shutil
from dataprocessing3 import data

os.environ["CUDA_VISIBLE_DEVICES"]="1"

try:
    shutil.rmtree("lineDataset_weightFile")
except OSError as e:
    #print ("Error: %s - %s." % (e.filename, e.strerror))
    k=1
    
    
os.mkdir("lineDataset_weightFile")
currentDir = os.getcwd()


path=input("Enter the path of your data file: ")
path1=input("Enter the path of your ground_truth file: ")
x_train,y_train=data(path,path1)
x_train=x_train
y_train=y_train

EPOCHS = 100
INIT_LR = 1e-3
BS = 32

#bbox_train = np_utils.to_categorical(bbox_train)

def baseline_model(inputs):
    # CONV => RELU => POOL
    x = Conv2D(64, (3, 3), padding="same")(inputs)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Dropout(0.4)(x)

    # CONV => RELU => POOL
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.4)(x)
# CONV => RELU => POOL
    x = Conv2D(32, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.4)(x)

    # CONV => RELU => POOL
    x = Conv2D(32, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.4)(x)   

    x = Conv2D(32, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.4)(x)

    x = Conv2D(32, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.4)(x)

   


    x = Conv2D(32, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.4)(x)

    

    # define a branch of output layers for the number of different
    # colors (i.e., red, black, blue, etc.)
    x = Flatten()(x)
    x = Dense(32)(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.50)(x)
    return x

inputShape = (600,400,3)
inputs = Input(shape=inputShape)
model = baseline_model(inputs)

regression = Dense(2, activation='relu', name='training_output')(model)
losses = {"training_output": "mae"}

lossWeights = {"training_output": 1.0}

designed_model = Model(inputs, [regression])
filepath = currentDir + "/lineDataset_weightFile"+ "/lineset-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

designed_model.compile(optimizer='adam', loss=losses, loss_weights=lossWeights, metrics=["accuracy"])
designed_model.summary()
designed_model.fit(x_train, {"training_output":y_train}, epochs=EPOCHS,batch_size=BS, verbose=1,callbacks= callbacks_list)

