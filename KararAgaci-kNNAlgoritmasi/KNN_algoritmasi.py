from sklearn.datasets import load_iris
iris = load_iris()
# bağımsız değişkenler
print (iris.feature_names)
# bağımlı değişkenleri 
print (iris.target_names)
#bağımlı değişkenleri sayısallaştırma
print (iris.target)
#bağımsız değişkenleri sayısallaştırma
print (iris.data)
X = iris.data
Y = iris.target


# egitim ve test verileri
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

print("Eğitim veri seti boyutu=",len(X_train))
print("Test veri seti boyutu=",len(X_test))

#model kurma
from sklearn.neighbors import KNeighborsClassifier
model =  KNeighborsClassifier ()
model.fit(X_train,Y_train)
Y_tahmin = model.predict(X_test)

#modelin hata durumunu görme
from sklearn.metrics import confusion_matrix
hata_matrisi = confusion_matrix(Y_test, Y_tahmin)
print(hata_matrisi)

#görselleştirme
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
index = ['setosa','versicolor','virginica'] 
columns = ['setosa','versicolor','virginica'] 
hata_goster = pd.DataFrame(hata_matrisi,columns,index)                      
plt.figure(figsize=(10,6))  
sns.heatmap(hata_goster, annot=True)