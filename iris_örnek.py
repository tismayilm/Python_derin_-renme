
#modülleri yükleyelim.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix


#Verimizi okuyalım.
veriler=pd.read_csv("ırıs.csv")

#Verimizi kolonlara ayıralım.
X=veriler.iloc[:,0:4].values # bağımlı değişken
Y=veriler.iloc[:,4].values # bağımsız değişken

 #Verimizi string ifadeden sayıya çevirelim
le1=LabelEncoder()
kolon=le1.fit_transform(Y)

#bağımsız değişkenimizi 0 ve 1'e çevirelim.
ohe=OneHotEncoder()
Y=ohe.fit_transform(Y[:,np.newaxis]).toarray()

#verimizi train ve test'e ayıralım.
Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,test_size=0.3,random_state=2)

#Verimizi ölçeklendirelim.
sc=StandardScaler()
Xo=sc.fit_transform(Xtrain)
Xto=sc.fit_transform(Xtest)


# modelimizi oluşturalım
model=Sequential()

model.add(Dense(8,activation="relu",input_dim=4))
model.add(Dense(8,activation="relu"))
model.add(Dense(3,activation="sigmoid"))

# modelimizi optimize edelim.
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

# modelimizi eğitelim
model.fit(Xo,Ytrain,epochs=200)

#modelimizi tahmin edelim.
y_pred=model.predict(Xto)
print(y_pred)


#modelimizi  grafikleştirelim.
plt.plot(y_pred,Ytest)
plt.scatter(y_pred,Ytest,color="red")

plt.show()






















































































