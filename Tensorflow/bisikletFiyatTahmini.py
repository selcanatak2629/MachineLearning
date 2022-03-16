# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 22:23:03 2022

@author: selca
"""

import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

dataFrame = pd.read_excel("bisiklet_fiyatlari.xlsx")

print(dataFrame.head())
print(sbn.pairplot(dataFrame))

x = dataFrame[["BisikletOzellik1","BisikletOzellik2"]].values
y= dataFrame["Fiyat"].values #bağımlı

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.33,random_state=15)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)

#scaling

scaler = MinMaxScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train)

model = Sequential()

#katman ve noron ekleme
model.add(Dense(4, activation= "relu"))
model.add(Dense(4, activation= "relu"))
model.add(Dense(4, activation= "relu"))

model.add(Dense(1))#cıktı
 
model.compile(optimizer="rmsprop",loss="mse" )

model.fit(x_train, y_train, epochs= 250)

loss = model.history.history["loss"]
print(sbn.lineplot(x=range(len(loss)), y=loss))

trainLoss = model.evaluate(x_train,y_train,verbose=0)
testLoss = model.evaluate(x_test,y_test,verbose=0)

print(trainLoss)
print(testLoss)

yeniBisikletozellikleri = [[1760,1758]]
yeniBisikletozellikleri = scaler.transform(yeniBisikletozellikleri)
print(model.predict(yeniBisikletozellikleri))

# model kayıt etme 
from tensorflow.keras.models import load_model

model.save("bisiklet_modeli.h5")

sonradanCagrilanmodel=load_model("bisiklet_modeli.h5")

print(sonradanCagrilanmodel.predict(yeniBisikletozellikleri))



