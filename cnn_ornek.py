

'''
Amaçımız fotoğraftaki nesneleri sınıflandırmak.
Algoritma: Evrişimli Sinir Ağları (Convolutional Neural Networks)
Epochs'u ne kadar arttırırsanız algoritmanız o kadar doğru çalışır.


'''


#modülleri yükleyelim

from __future__ import print_function
import keras 
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout,Conv2D,MaxPooling2D,Activation
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import matplotlib.pyplot as plt

batch_size = 32 # Her iterasyonda 32 fotoğraf alınır.
num_classes = 10 # CIFAR10 veri setinde 10 sınıf bulunmakta.
epochs = 1 # 200 epoch ile eğitim yapılacaktır.


#verimizi karıştırıp train ve teste sokuyoruz.
(Xtrain,Ytrain),(Xtest,Ytest)=cifar10.load_data()

print("Xtrain_shape:" ,Xtrain.shape)
print(Xtrain.shape[0],"Eğitim Sayısı")
print(Xtest.shape[0],"Test Sayısı")

#Sınıfımızda (ikili)binary'ye dönüştürülür.Ve Ytrain,Ytest to_categorical fonksiyonu ile one-hot-encoding yapalır.

Ytrain=to_categorical(Ytrain,num_classes)
Ytest=to_categorical(Ytest,num_classes) 


#Gerekli İşlemleri Yaptıktan sonra modelimizi oluşturmaya başlayalım.

model=Sequential()

# 32 adet 3x3 boyutunda filtereler oluşturulur ve modele eklenir.
# "Padding" fotoğrafa çervçeve ekler ve çıkış boyutunun giriş boyutuna eşit olması sağlanır.
# Relu Activasyon Fonsiyinumuzu Ekleyelim.

model.add(Conv2D(32,(3,3),padding="same", input_shape=Xtrain.shape[1:],activation="relu"))

# 32 adet 3x3 boyutunda filterelerden oluşan katmanımızı modelimize ekliyoruz:
model.add(Conv2D(32,(3,3),activation="relu"))

# 2x2 boyutunda çerçeveden oluşan MaxPooling katmanımızı ekliyoruz.
model.add(MaxPooling2D(2,2))

# Rastgele olacak şekilde nöronların %25'ini kapatıyoruz: (Eğitim sırasındaki ezberlemeyi önlemek için.)
model.add(Dropout(0.25))


# 64 adet 3x3 boyutunda filterelerden oluşan katmanımızı modelimize ekliyoruz:
# "Padding" fotoğrafa çervçeve ekliyoruz.
# Relu Activasyon Fonsiyinumuzu Ekleyelim.
model.add(Conv2D(64,(3,3),padding="same",activation="relu"))

# 64 adet 3x3 boyutunda filterelerden oluşan katmanımızı modelimize ekliyoruz:
model.add(Conv2D(64,(3,3),activation="relu"))

# 2x2 boyutunda çerçeveden oluşan MaxPooling katmanımızı ekliyoruz.
model.add(MaxPooling2D(2,2))

# Rastgele olacak şekilde nöronların %25'ini kapatıyoruz: (Eğitim sırasındaki ezberlemeyi önlemek için.)
model.add(Dropout(0.25))


# Görselimizi 2 boyutluya çevirdik sonra 1 boyutlu vectöre çevirelim.
model.add(Flatten())

# 512 nörondan oluşan modelimizi ekliyoruz.
#Activasyon fonksiyonumuzu veriyoruz.
model.add(Dense(512,activation="relu"))

# Rastgele olacak şekilde nöronların %50'sini kapatıyoruz:
model.add(Dropout(0.5))

# 10 sınıfımızı temsil edecek 10 nöronumuzu modelimize ekliyoruz:
model.add(Dense(num_classes))

# Sınıfların olasılıklarını hesaplamak için "Softmax" fonksiyonumuzu ekliyoruz:
model.add(Activation('softmax'))

# Modelimizi oluşturduk.Şimdi modelimizi optimize edelim.
# Sınıflandırma yapacağımız için "categorical_crossentropy" fonksiyonunu kullanıyoruz.
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['accuracy'])


 # Canlı veri arttırmak için ayarlarımızı yapıyoruz:
canlı_data = ImageDataGenerator(
                                 featurewise_center=False,
                                 samplewise_center=False,
                                 featurewise_std_normalization=False,
                                 samplewise_std_normalization=False,
                                 zca_whitening=False,
                                 rotation_range=0,
                                 width_shift_range=0.1,  # Görüntüleri rasgele olarak yatay olarak kaydırın.
                                 height_shift_range=0.1,  # Görüntüleri rasgele olarak dikey olarak kaydırın.
                                 horizontal_flip=True,  # Fotoğrafı yatay düzlemde rastgele çevirme.
                                 vertical_flip=False)


# Verimizi oluşturmak için  Hesaplıyoruz.
canlı_data.fit(Xtrain)  


# Verimizi oluşturduktan sonra fit_generator fonksiyonu ile verimizi eğitiyoruz.

model.fit_generator(canlı_data.flow(Xtrain,Ytrain,
                                    batch_size=batch_size),
                                    steps_per_epoch=int(np.ceil(Xtrain.shape[0] / float(batch_size))),
                                    verbose=1,
                                    epochs=epochs,
                                    validation_data=(Xtest,Ytest),
                                    workers=4)



scores = model.evaluate(Xtest,Ytest, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])



