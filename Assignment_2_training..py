import keras
from keras.utils import np_utils, to_categorical
from keras import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from keras.callbacks import ModelCheckpoint
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from multi_class_seperation_processing import multi_class_seperation_processing

seed = 7
np.random.seed(seed)

epoch = 1
batch_size = 100


(X_train), (X_test), length_train, width_train, color_train,angle_train,  length_test, width_test,  color_test, angle_test = multi_class_seperation_processing("./dataset/")

angle_train = np_utils.to_categorical(angle_train)
angle_test = np_utils.to_categorical(angle_test)

# Model
X = Input(shape=(28, 28, 3), name='line')

conv1 = Conv2D(filters=16, kernel_size=[3, 3], strides=1, padding='same', activation='relu', data_format='channels_last')(X)
norm1 = BatchNormalization()(conv1)
maxpool1 = MaxPooling2D(pool_size=[3, 3], strides=1, padding='same')(norm1)
drop1 = Dropout(0.15)(maxpool1)
conv2 = Conv2D(filters=32, kernel_size=[3, 3], strides=1, padding='same', activation='relu')(drop1)
norm2 = BatchNormalization()(conv2)
maxpool2 = MaxPooling2D(pool_size=[2, 2], strides=1, padding='same')(norm2)
drop2 = Dropout(0.15)(maxpool2)
conv3 = Conv2D(filters=32, kernel_size=[3, 3], strides=1, padding='same', activation='relu')(drop2)
norm3 = BatchNormalization()(conv3)
maxpool3 = MaxPooling2D(pool_size=[2, 2], strides=1, padding='same')(norm3)
drop3 = Dropout(0.15)(maxpool3)
feature = Flatten()(drop3)
dense1 = Dense(1024)(feature)
activation = Activation('relu')(dense1)
norm = BatchNormalization()(activation)
drop = Dropout(0.5)(norm)


length_classification = Dense(1, activation='sigmoid', name='length')(drop)

width_classification = Dense(1, activation='sigmoid', name='width')(drop)

angle_classification = Dense(12, activation='softmax', name='angle')(drop)

color_classification = Dense(1, activation='sigmoid', name='color')(drop)


designed_model = Model([X], [length_classification, width_classification, angle_classification, color_classification])

lossNames = ["loss", "length_loss", "width_loss", "angle_loss", "color_loss"]
lossWeights = {"line_output": 1.0, "width_output": 1.0, "angle_output": 1.0, "color_output": 1.0}

designed_model.compile(optimizer=Adam(lr=0.001, decay=1e-5),
                       loss={'length': 'binary_crossentropy',
                             'width': 'binary_crossentropy',
                             'angle': 'categorical_crossentropy',
                             'color': 'binary_crossentropy'},
                       metrics=['accuracy'])

designed_model.summary()

filepath = "/home/mahesh/PycharmProjects/hello/lineset2-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history = designed_model.fit(x=X_train, y=[length_train, width_train, angle_train, color_train], batch_size=batch_size, epochs=epoch, verbose=1, validation_split=0.2, callbacks= callbacks_list)

scores = designed_model.evaluate(X_test, [length_test, width_test, angle_test, color_test], batch_size=batch_size, verbose=1)


plt.style.use("ggplot")
(fig, ax) = plt.subplots(3, 1, figsize=(13, 13))

# loop over the loss names
for (i, l) in enumerate(lossNames):
	# plot the loss for both the training and validation data
	title = "Loss for {}".format(l) if l != "loss" else "Total loss"
	ax[i].set_title(title)
	ax[i].set_xlabel("Epoch #")
	ax[i].set_ylabel("Loss")
	ax[i].plot(np.arange(0, epoch), history.history[l], label=l)
	ax[i].plot(np.arange(0, epoch), history.history["val_" + l],
		label="val_" + l)
	ax[i].legend()

plt.show()
# save the losses figure and create a new figure for the accuracies
plt.tight_layout()
plt.savefig("{}_losses.png")
# plt.close()

# print("Total Loss:", scores[0])
# print("Length Loss:", scores[1])
# print("Width Loss:", scores[2])
# print("Angle Loss:", scores[3])
# print("Color Loss:", scores[4])
#



