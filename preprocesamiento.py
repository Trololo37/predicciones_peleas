#metodos de entrenamiento que mas me gustaron hasta ahora
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

#visualizacion de la informacion
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#ayudas
import pandas as pd
import torch
import csv
"""import keras"""

#importacion de todos los datos
from data import absolutos as data

with open("data/data.csv", newline='') as datacsv:
    DATA_CSV = csv.reader(datacsv)
    i = 0
    for row in DATA_CSV:
        if(i>5):
            break
        print(', '.join(row))
        i+=1

