from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import random
import os
import cv2
from PIL import Image
#import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Input, UpSampling2D, BatchNormalization
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split


seed = 7
np.random.seed(seed)

data_path = "./Assignment3/Q2/Data"
mask_path = "./Assignment3/Q2/Mask"


def get_data(data_path, mask_path):
    data = []
    for frame in os.listdir(data_path):
        im_data = cv2.imread(os.path.join(data_path, frame))
        #im_data = cv2.cvtColor(im_data, cv2.COLOR_BGR2GRAY)
        im_data = cv2.resize(im_data, (256, 256))
        data_array = np.array(im_data)
        data.append(data_array)

    data = np.asarray(data)
    #data = data/255

    mask = []
    for f in os.listdir(mask_path):
        im_mask = cv2.imread(os.path.join(mask_path, f),0)
        #im_mask = cv2.cvtColor(im_mask, cv2.COLOR_BGR2GRAY)
        im_mask = cv2.resize(im_mask, (256, 256))
        mask_array = np.array(im_mask)
        mask.append(mask_array)

    mask = np.asarray(mask)
    #mask = mask/255

    return data, mask


data, mask = get_data(data_path, mask_path)
print(data.shape)
print(mask.shape)
data_train, data_test, mask_train, mask_test = train_test_split(data, mask, test_size=0.1)

def image_generator(x_train,y_train, batch_size = 32):

    count= -batch_size
    y_train=np.expand_dims(y_train, axis=3)
    while True:
        # Select files (paths/indices) for the batch
        count+=batch_size
        if(count>=9000-batch_size):
            count=0
        batch_input = []
        batch_output = []

        # Read in each input, perform preprocessing and get labels
        for i in range(batch_size):
            input1 = x_train[i+count]
            output = y_train[i+count]

            # input = preprocess_input(image=input)
            batch_input += [ input1 ]
            batch_output += [ output ]
        # Return a tuple of (input,output) to feed the network
        batch_x = np.array( batch_input )
        batch_y = np.array( batch_output )
        #print(batch_x.shape,batch_y.shape)
        yield( batch_x, batch_y )


def unet(pretrained_weights=None, input_size=(256,256,3)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    drop1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(drop1)

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    #conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    drop2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    #conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    #conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    #conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    #conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    drop6 = Dropout(0.5)(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    #conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    drop7 = Dropout(0.5)(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    #conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    drop8 = Dropout(0.5)(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    drop9 = Dropout(0.5)(conv9)
    conv10 = Conv2D(3, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid', padding='same')(conv10)

    model = Model(inputs, conv10)

    model.compile(optimizer=Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

model = unet()
model.summary()

batch_size = 16

tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./Graph_UNET', histogram_freq=0, write_graph=True, write_images=True)

#data_train = np.squeeze(data_train, axis=3)
#mask_train = np.squeeze(mask_train, axis=3)

trainGen = image_generator(x_train=data_train, y_train=mask_train, batch_size=batch_size)

model.fit_generator(trainGen, epochs=10, steps_per_epoch=9000/batch_size, callbacks=[tbCallBack])
print(data_test.shape,mask_test.shape)
mask_test = np.expand_dims(mask_test, axis=3)
score = model.evaluate(data_test, mask_test)
print(score)

#data_test = np.expand_dims(data_test, axis=3)
y_pred = model.predict(data_test)
print('y_pred.shape=' + str(y_pred.shape))
print(y_pred[0])
#for i in range(10):
#    cv2.imwrite('output3/'+str(i)+'_pred.jpg', y_pred[i]*255)
#    cv2.imwrite('output3/'+str(i)+'_input.jpg', data_test[i]*255)
#    cv2.imwrite('output3/'+str(i)+'_gt.jpg', mask_test[i]*255)

