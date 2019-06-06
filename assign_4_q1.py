import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam
import pickle
from keras.layers import Flatten,Dropout,Activation
from keras.preprocessing.sequence import pad_sequences

np.random.seed(7)
look_back = 2
# convert an array of values into a dataset mat
seq=[]
# k=[]
def ap(a,n,d):
	for i in range(n):
		p=a+(i-1)*d
		seq.append(p)
	return seq
k=np.asarray(ap(0,100,5))

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

k = np.reshape(k,(k.shape[0],1))
# print(k.shape)
# print(k)
# [m,n]=create_dataset(k, look_back=2)
#
# print(m.shape)
# print(m)
# print(n.shape)
# print(n)
# trainX = m
# trainY = n
# split into train and test sets
train_size = int(len(k) * 0.67)
test_size = len(k) - train_size
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
k = scaler.fit_transform(k)
train, test = k[0:train_size], k[train_size:len(k)]

trainX, trainY = create_dataset(train, look_back=2)
testX, testY = create_dataset(test, look_back=2)

trainX=np.reshape(trainX,(trainX.shape[0],trainX.shape[1],1)).astype('float32')
testX = np.reshape(testX,(testX.shape[0],testX.shape[1],1)).astype('float32')
# trainY = np.reshape(trainY,(trainY.shape[0],1)).astype('float32')
# testY = np.reshape(testY,(testY.shape[0],1)).astype('float32')
# print(trainX)
# print("******************************")
# print(trainY.shape)
# print("******************************")
# print(trainX)
# print(trainY)
# print(testX)
# print(testY)



pickle_in = open("/home/mahesh/PycharmProjects/hello/Deep Learning/assign_4/Assignment4/ap_5.pkl","rb")
example_dict = pickle.load(pickle_in)
pk=np.asarray(example_dict)

# print(pk.shape)

test2=[]

for i in range(10):
	test1=np.asarray(pk[i,0])
	test2.append(test1)

test2=np.asarray(test2)
# print(test2)

test3=np.asarray(pk[:,1])
# print(test3.shape)


EPOCHS = 1
INIT_LR = 1e-3

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

# print(trainX.shape)
# print(trainX)
# print(trainY.shape)
# print(trainY)

# look_back=1
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(16,input_shape=(trainX.shape[1],trainX.shape[2]),return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(16))
# model.add(LSTM(16,batch_input_shape=(None,1),return_sequences=True))
# model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(1))
model.add(Activation('relu'))
model.compile(loss='mean_squared_error', optimizer=opt)
model.summary()
model.fit(trainX,trainY,epochs=EPOCHS, batch_size=1, verbose=1)

# X = [[100,105]]
# # print(np.asarray(X).shape)
# X = scaler.fit_transform(X)
# x = np.reshape(X,(np.asarray(X).shape[0],np.asarray(X).shape[1],1)).astype('float32')
# p1 = model.predict(x, verbose=0)
# print(p1)
# # p = p.tolist()
# p = scaler.inverse_transform(p1)
# print(p)
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
print("******************************")
print(trainPredict.shape)
print("******************************")
print(testPredict.shape)
print("******************************")

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# print("******************************")
# print(trainPredict)
# print("******************************")
# print(testPredict)
# print("******************************")

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = np.empty_like(k)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(k)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(k)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(k))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
