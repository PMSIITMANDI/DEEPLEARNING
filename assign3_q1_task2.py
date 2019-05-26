import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.utils import np_utils
from keras import regularizers
from keras.layers import Input
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Lambda
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
import tensorflow as tf
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.utils import np_utils
from keras import regularizers
import h5py
from keras.callbacks import ModelCheckpoint
from keras.losses import binary_crossentropy
from keras.losses import sparse_categorical_crossentropy
import glob

import os
# import numpy as np
path = "/home/mahesh/PycharmProjects/hello/Deep Learning/assignment_3/Assignment3/Q1/Four_Slap_Fingerprint/Ground_truth"
x1_1 = []
x1_2 = []
x1_3 = []
x1_4 = []

x2_1 = []
x2_2 = []
x2_3 = []
x2_4 = []

y1_1 = []
y1_2 = []
y1_3 = []
y1_4 = []

y2_1 = []
y2_2 = []
y2_3 = []
y2_4 = []

allGroundTruthTextFiles = os.listdir(path)

for f in allGroundTruthTextFiles:
    file = open(path + "/" + str(f))

    # print (file.read())
    line1 = file.readline()
    line2 = file.readline()
    line3 = file.readline()
    line4 = file.readline()

    splitLine = line1.split(",")
    x1_1.append(splitLine[0])
    y1_1.append(splitLine[1])
    x2_1.append(splitLine[2])
    y2_1.append(splitLine[3])

    splitLine = line2.split(",")
    x1_2.append(splitLine[0])
    y1_2.append(splitLine[1])
    x2_2.append(splitLine[2])
    y2_2.append(splitLine[3])

    splitLine = line3.split(",")
    x1_3.append(splitLine[0])
    y1_3.append(splitLine[1])
    x2_3.append(splitLine[2])
    y2_3.append(splitLine[3])

    splitLine = line4.split(",")
    x1_4.append(splitLine[0])
    y1_4.append(splitLine[1])
    x2_4.append(splitLine[2])
    y2_4.append(splitLine[3])
finalMatrix = np.column_stack((np.asarray(x1_1), np.asarray(y1_1), np.asarray(x2_1), np.asarray(y2_1), np.asarray(x1_2),
                               np.asarray(y1_2), np.asarray(x2_2), np.asarray(y2_2), np.asarray(x1_3), np.asarray(y1_3),
                               np.asarray(x2_3), np.asarray(y2_3), np.asarray(x1_4), np.asarray(y1_4), np.asarray(x2_4),
                               np.asarray(y2_4)))

# # load data from the path specified by the user
#
image = []
path1 = './Assignment3/Q1/Four_Slap_Fingerprint/Data/*.jpg'
for file in glob.glob(path1):
    I = cv2.imread(file)
    #I=cv2.resize(I,(480,640))
    img = np.asarray(I)
    image.append(img)
image = np.asarray(image)
print(image.shape)
X_train, X_test, bbox_train, bbox_test = train_test_split(image, finalMatrix, test_size=0.25, random_state=42)
X_train=X_train/255
EPOCHS = 10
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (1572,1672)
def basel_model(inputs):
    # CONV => RELU => POOL
    x = Conv2D(128, (3, 3), padding="same")(inputs)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Dropout(0.25)(x)
    # CONV => RELU => POOL
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    # CONV => RELU => POOL
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    # CONV => RELU => POOL
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    # CONV => RELU => POOL
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    # CONV => RELU => POOL
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    # CONV => RELU => POOL
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    # CONV => RELU => POOL
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    # CONV => RELU => POOL
    x = Conv2D(32, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.25)(x)
    # CONV => RELU => POOL
    x = Conv2D(32, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    #x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(32, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    #x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(32, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    #x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(32, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    #x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)


    x = Conv2D(32, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    
    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.45)(x)

    return x

inputShape = (X_train.shape[1],X_train.shape[2],X_train.shape[3])
inputs = Input(shape=inputShape)
model = basel_model(inputs)

head_regression = Dense(16, activation='relu', name='reg_output')(model)
losses = {"reg_output": "mae"}

lossWeights = {"reg_output": 1.0}

finger_localocalisation_model = Model(inputs, [head_regression])

#callback_1 = keras.callbacks.TensorBoard(log_dir='./graphs', histogram_freq=0, write_graph=True, write_images=True)
filepath="./Assignment3/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
finger_localocalisation_model.compile(optimizer='nadam', loss=losses, loss_weights=lossWeights, metrics=["accuracy"])
finger_localocalisation_model.summary()
finger_localocalisation_model.fit(X_train, {"reg_output":bbox_train}, epochs=EPOCHS,batch_size=BS, verbose=1)


