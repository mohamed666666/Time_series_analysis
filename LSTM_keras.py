#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 09:15:24 2020

@author: v
"""
def scaling_normalizaing(x):
    return (x-np.min(x))/( np.max(x)-np.min(x))


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


dataset=pd.read_csv("Google_Stock_Price_Train.csv")

training_set=dataset.iloc[:,1:2].values
#feature scaling  .
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
sc_trianing=sc.fit_transform(training_set)


#redesign our dataset to be as ({60 X<t>} so we gave 60   time steps,  𝑇𝑥=60 )
#and output is one day later 
x_train=[]
y_train=[]
for i in range(60,1258):
    x_train.append(sc_trianing[i-60:i])
    y_train.append(sc_trianing[i])

x_train=np.array(x_train)
y_train=np.array(y_train)


#reshape xtrain because thates the keras model input shape(m,T𝑥,n𝑥) bs a7a ya3ni fe eltutorial ma3mlsh keda habb 
#keda
# we must stack dimntions of the output = number of what we wanna predict in this ex its one because we predict open 
x_train=x_train.reshape((1198, 60,1))



from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

model=Sequential()
#first LSTM layer
model.add(LSTM(units=50,return_sequences=True,input_shape=(60,1)))
model.add(Dropout(0.2))


#secound layer 
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))

#third LSTM layer

model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))

#fourth LSTM layer 
model.add(LSTM(units=50))
model.add(Dropout(0.2))


# output layer 

model.add(Dense(units=1))


model.compile(optimizer='adam',loss='mean_squared_error')

model.fit(x_train,y_train,epochs=90,batch_size=32)


#load test data set

dataset_test=pd.read_csv("Google_Stock_Price_Test.csv")

test_set=dataset_test.iloc[:,1].values


#concate all data to make predictions 

total_data=pd.concat((dataset["Open"],dataset_test["Open"]),axis=0)
inputs=total_data.iloc[len(total_data)-len(dataset_test)-60:].values

sc_test=sc.transform(inputs)

x_test=[]
for i in range(60,80):
    x_test.append(sc_test[i-60:i])

x_test=np.array(x_test)
x_test=x_test.reshape((20, 60,1))



preds=model.predict(x_test)

predicted_stock_price=sc.inverse_transform(preds)


#visualizing
plt.plot(test_set,color='red',label="real data ")  
plt.plot(predicted_stock_price,color='blue',label="Predicted data ")  
plt.legend()
plt.show()













