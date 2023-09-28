from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
#import matplotlib.pyplot as plt
#import numpy
#from keras.datasets import mnist
import pandas as pd
#import torch




x_train = pd.read_csv("x_train.csv")
y_train = pd.read_csv("y_train.csv")

x_test = pd.read_csv("x_test.csv")
y_test = pd.read_csv("y_test.csv")


resultados = [0 for i in range(5,31,5)]
i=0
for k in range(5,31,5):
    knn_clf = KNeighborsClassifier(n_neighbors=k, metric='minkowski', n_jobs=-1)
    knn_clf.fit(x_train, y_train)
    yhat=knn_clf.predict(x_test)
    resultados[i]=accuracy_score(y_test, yhat)
    print("vecinos {}   precision {}".format(k, resultados[i]))
    i+=1