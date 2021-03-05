# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 16:35:19 2020

@author: Turhan
"""
import pandas as pd
#Veri Hazırlama
sutun = ["anadil","egitmen","kurs","donemi"
            ,"sinifbuyuklugu","sinifozelligi"
]
veri = pd.read_csv("tae.data", names=sutun)
girdiler = veri.iloc[:,0:-1:]        
hedef = veri.iloc[:,-1:]   
from sklearn.model_selection import train_test_split
X_egitim, X_test, y_egitim, y_test = train_test_split(girdiler, hedef, test_size=0.30,
random_state=45)
#Modelleme
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_egitim, y_egitim)
#Model Değerlendirme
tahmin_test = pd.DataFrame(dtc.predict(X_test))
tahmin_egitim = pd.DataFrame(dtc.predict(X_egitim))
from sklearn.metrics import confusion_matrix
cm_egitim = confusion_matrix(y_egitim, tahmin_egitim)
cm_test = confusion_matrix(y_test, tahmin_test)
print(cm_egitim)
print(cm_test)
from sklearn.metrics import accuracy_score
as_egitim = accuracy_score(y_egitim, tahmin_egitim)
as_test = accuracy_score(y_test, tahmin_test)
print("Eğitim Doğruluk Oranı >>>",as_egitim)
print("Test Doğruluk Oranı >>>",as_test)

#Konuşlandırma
sutun=[]
anadil=float(input("Ana Dilini Giriniz(inligizce ise 1 değilse 2): "))
sutun.append(anadil)
egitmen=float(input("Egitmen Giriniz(kategorik, 25 kategori): "))
sutun.append(egitmen)
kurs=float(input("Kurs Giriniz(kategorik, 26 kategori): "))
sutun.append(kurs)
donem=float(input("Dönem Giriniz(yaz ise 1 normal ise 2): "))
sutun.append(donem)
sinifbuyuklugu=float(input("Sınıf Büyüklüğünü Giriniz(sayısal): "))
sutun.append(sinifbuyuklugu)
import numpy as np
dizi=np.asarray(sutun)
dizi_rs=dizi.reshape(1,-1)
tahmin=dtc.predict(dizi_rs)
print("Sınınf Özelliği Başarı Düzeyi(1=düşük, 2=orta, 3=yüksek): ", tahmin)








