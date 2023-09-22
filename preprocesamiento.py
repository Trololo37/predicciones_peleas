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
"""import keras"""

"""#importacion de todos los datos
import csv
import data

with open("data/data.csv", newline='') as archivo:
    informacion = csv.reader(archivo)
    for fila in informacion:
        print(', '.join(fila))
print(informacion)"""