# coding=utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

'''
from Avik-Jain/100-Days-Of-ML-Code
'''

def logistic_regression():
    #Step 1 data pre-processing
    #import dataset
    dataset = pd.read_csv('Social_Network_Ads.csv')
    X = dataset.iloc[:, [2,3]].values
    Y = dataset.iloc[:, 4].values

    #Splitting the dataset into the Training set and Test set
    X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.25, random_state = 0)

    #Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    #Step 2 Logistic Regression Model
    #Fitting Logistic Regression to the Training set
    classifier = LogisticRegression()
    classifier.fit(X_train, Y_train)

    #Step 3 Predection
    Y_pred = classifier.predict(X_test)
    print(Y_pred)

    #Step 4 Evaluating The Predection
    #Making the Confusion Matrix
    cm = confusion_matrix(Y_test, Y_pred)
    print(cm)

if __name__ == "__main__":
    logistic_regression()
