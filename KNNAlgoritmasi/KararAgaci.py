#sklearn.datasets kütüphanesinden load_iris clasını import ediyoruz
from sklearn.datasets import load_iris
#değişkene atıyoruz ki bunun üzerinden işlem yapıcaz
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

# modeli eğitmek için verilerimizi eğitim ve test için bölüyoruz
#tarinler model kurmak için testler test etmek için sonuc verır
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)
print("Eğitim veri seti boyutu=",len(X_train))
print("Test veri seti boyutu=",len(X_test))

#Model Kurma

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
#modeli eğitme
model.fit(X_train,Y_train) 
#modeli test etme
Y_tahmin = model.predict(X_test)

# hata oranını modelin dogrululugunu gormek ıstıyruz
#Tahmin-Test Sonuçlarını Karşılaştırma 
from sklearn.metrics import confusion_matrix
hata_matrisi = confusion_matrix(Y_test, Y_tahmin)
print(hata_matrisi)

#sonucumuzu görselleştirme
import seaborn as sns
import pandas as pd #veri analizi
import matplotlib.pyplot as plt #görselleştime
index = ['setosa','versicolor','virginica'] 
columns = ['setosa','versicolor','virginica'] 
hata_goster = pd.DataFrame(hata_matrisi,columns,index)                      
plt.figure(figsize=(10,6))  
sns.heatmap(hata_goster, annot=True)