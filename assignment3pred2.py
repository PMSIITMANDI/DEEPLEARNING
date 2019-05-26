#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 23:56:37 2019

@author: pushap
"""

import glob
import os
from sklearn.model_selection import train_test_split

import keras
#from keras.models import Sequential
#from keras.layers import Dense,Conv2D,Dropout,Flatten,MaxPooling2D
from keras.models import Model
#from keras.layers.normalization import BatchNormalization
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
    shutil.rmtree("coordinate_file")
except OSError as e:
    #print ("Error: %s - %s." % (e.filename, e.strerror))
    k=1
    
    
os.mkdir("coordinate_file")

currentDir = os.getcwd()

weightFiles = os.listdir("./lineDataset_weightFile/")
weightFiles.sort()
weightFilepath = currentDir+"/lineDataset_weightFile/"+weightFiles[-1]
print(weightFilepath)
path=raw_input("Enter the path of your data file: ")
path1=raw_input("Enter the path of your ground_truth file: ")
x,y=data(path,path1)
x=x
y=y
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
print(X_train.shape)
print(y_test[3])
print(X_test[3])

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

designed_model = Model(inputs,[regression])
designed_model.load_weights(weightFilepath)
designed_model.compile(optimizer='adam', loss=losses, metrics=["accuracy"])
designed_model.summary()
z=designed_model.evaluate(X_test,y_test,batch_size=32,verbose=1)
pre=designed_model.predict(X_test)
print("X=%s, predicted=%s" %(X_test[3],pre[3]))
