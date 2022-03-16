# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 13:22:12 2022

@author: selca
"""
import numpy as np
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt

dataFrame = pd.read_excel("merc.xlsx")

print(dataFrame.head())
#tüm verileri daha genel görürsün
print(dataFrame.describe())
#butun veriler tam eksik veri yok
print(dataFrame.isnull().sum())

plt.figure(figsize=(7,5))
sbn.distplot(dataFrame["price"])
sbn.countplot(dataFrame["year"])
print(dataFrame.corr()["price"].sort_values)

sbn.scatterplot(x="mileage", y="price", data=dataFrame)

#false yani fiyatı yüksekten düşüğe göre
print(dataFrame.sort_values("price",ascending= False))
#true yani fiyatı düşükten yükseğe göre
print(dataFrame.sort_values("price",ascending= True).head(20))
print(len(dataFrame))
print(len(dataFrame)*0.01)

yuzdeDoksanDokuzDf = dataFrame.sort_values("price",ascending=False).iloc[131:]
print(yuzdeDoksanDokuzDf.describe())

plt.figure(figsize=(7,5)) 
sbn.distplot(yuzdeDoksanDokuzDf["price"])

print(dataFrame.groupby("year").mean()["price"])
print(dataFrame[dataFrame.year != 1970].groupby("year").mean()["price"])

#düzenediğimiz verileri ele alıyoruz
dataFrame = yuzdeDoksanDokuzDf

dataFrame=dataFrame[dataFrame.year != 1970]

print("----------------------------------------------")
print(dataFrame.groupby("year").mean()["price"])
#dataFrame = dataFrame.drop("transmission",axis=1)

y = dataFrame["price"].values
x = dataFrame.drop("transmission",axis=1).values


from sklearn.model_selection import train_test_split

x_train , x_test, y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=10)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from tensorflow.keras.models import Sequential # model olusturma
from tensorflow.keras.layers import Dense # katman olursturma

# verilerimizi ve ozellıklerımızı gosterır
print(x_train.shape)

model = Sequential()

model.add(Dense(12, activation="relu"))
model.add(Dense(12, activation="relu"))
model.add(Dense(12, activation="relu"))
model.add(Dense(12, activation="relu"))

model.add(Dense(1))

model.compile(optimizer="adam",loss="mse")

model.fit(x=x_train, y=y_train, validation_data=(x_test,y_test),batch_size=250 ,epochs=300)


kayipVerisi=pd.DataFrame(model.history.history)
print(kayipVerisi.head())
print(kayipVerisi.plot())











