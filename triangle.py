import math
import pickle
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense,Dropout,Activation
from keras.optimizers import Adam
from keras import losses

def triangle(length, amplitude):
    section = length // 4
    for direction in (1, -1):
        for i in range(section):
            yield i * (amplitude / section) * direction
        for i in range(section):
            yield (amplitude - (i * (amplitude / section))) * direction


X = list(triangle(19,1))
train_data = np.asarray(X)

with open('triangle_tests.pkl', 'rb') as f:
    test_data = pickle.load(f)
test_data = np.asarray(test_data)

X_test_data = test_data[3,0]
#X_test = np.asarray(X_test)
Y_test_data = test_data[3,1]
#print(X_test.shape)
X_test_data = np.append(X_test_data,Y_test_data)
# train_data = np.asarray(train_data)
print(X_test_data.shape)
# X_test =[]
#
X_train =[]
Y_train = []
X_test = []
Y_test = []

#
# for i in range(10-sequence_length[i]):
#     test = test_data[i:(sequence_length-1)+i]
#     X_test.append(test)
#
#
print(len(train_data))
#print(len(train_data)-sequence_length)
c = 7

for i in range(len(train_data)-c):
    X = train_data[i:(c-1)+i]
    X_train.append(X)
    Y = train_data[i+c]
    Y_train.append(Y)
X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)
#X_test = np.asarray(X_test)

for i in range(len(X_test_data)-c):
    X_1 = X_test_data[i:(c-1)+i]
    X_test.append(X_1)
    Y_1 = X_test_data[i+c]
    Y_test.append(Y_1)
X_test = np.asarray(X_test)
Y_test = np.asarray(Y_test)
print(X_test.shape)
print(Y_test.shape)
#print(X_test.shape)


X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
Y_train = np.reshape(Y_train,(Y_train.shape[0],1))
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
Y_test = np.reshape(Y_test,(Y_test.shape[0],1))



model = Sequential()
model.add(LSTM(32, return_sequences=True, input_shape=(X_train.shape[1],1)))
model.add(LSTM(16,return_sequences=True))
#model.add(LSTM(8,return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(8))
model.add(Dropout(0.5))
#model.add(LSTM(5))
model.add(Dense(8))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss= 'mae' , optimizer= 'adam', metrics=['acc'])
print(model.summary())
#
#
model.fit(X_train, Y_train, batch_size=32, epochs=10)
#
y_hat = model.predict(X_test)
#
print(y_hat)
plt.plot(y_hat)
plt.show()