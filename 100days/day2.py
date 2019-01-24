# coding=utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

'''from Avik-Jain/100-Days-Of-ML-Code'''

def linear_regression():
    #Data Preprocessing
    dataset = pd.read_csv('studentscores.csv')
    X = dataset.iloc[ : , : 1].values
    Y = dataset.iloc[ : , 1].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

    #Fitting Simple Linear Regression Model to the training set
    regressor = LinearRegression()
    regressor = regressor.fit(X_train, Y_train)

    #Predecting the Result
    Y_pred = regressor.predict(X_test)

    #Visualization
    #Visualising the Training results
    plt.scatter(X_train, Y_train, color = 'red')
    plt.plot(X_train, regressor.predict(X_train), color = 'blue')
    
    #Visualizing the test results
    plt.scatter(X_test, Y_test, color = 'red')
    plt.plot(X_test, regressor.predict(X_test), color = 'blue')

if __name__ == "__main__":
    linear_regression()
