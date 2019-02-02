# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

def KNN():
    #Importing the dataset
    dataset = pd.read_csv('Social_Network_Ads.csv')
    X = dataset.iloc[:, [2, 3]].values
    Y = dataset.iloc[:, 4].values

    #Splitting the dataset into the Training set and Test set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

    #Feature Scaling
    sc = StandardScaler()
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)

    #Fitting K-NN to the Training set
    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    classifier.fit(X_train, Y_train)

    #Predicting the Test set results
    Y_pred = classifier.predict(X_test)

    #Making the Confusion Matrix
    cm = confusion_matrix(Y_test, Y_pred)