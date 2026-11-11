import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

pois_train = pd.read_csv('mushroom_poisonous_train.csv')
pois_test = pd.read_csv('mushroom_poisonous_test.csv')
habi_train = pd.read_csv('mushroom_habitat_train.csv')
habi_test = pd.read_csv('mushroom_habitat_test.csv')

def distance_func(df, i, j):
    return df.iloc[i].ne(df.iloc[j]).sum()

def hamming(x, y):
    if x == y: return 0
    return 1

def k_folds(df, k):
    return [fold for fold in np.array_split(df, k)]

print(distance_func(pois_train, 0, 1))
