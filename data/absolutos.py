#importacion de todos los datos
import csv


with open("data/data.csv", newline='') as archivo1:
    DATA_CSV = csv.reader(archivo1)

with open("data/raw_fighter_details.csv", newline='') as archivo2:
    RAW_FIGHTERS = csv.reader(archivo2)

with open("data/raw_total_fight_data.csv", newline='') as archivo3:
    RAW_FIGHTS = csv.reader(archivo3)

with open("data/preprocessed_data.csv", newline='') as archivo4:
    PRE_IN_CSV = csv.reader(archivo4)