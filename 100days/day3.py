# coding=utf-8

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def multiple_linear_regression():
    #Step 1: Data Preprocessing
    dataset = pd.read_csv('50_Startups.csv')
    X = dataset.iloc[ : , :-1].values
    Y = dataset.iloc[ : , 4 ].values

    labelencoder = LabelEncoder()
    X[ : , 3] = labelencoder.fit_transform(X[ : , 3])
    #print(X)
    onehotencoder = OneHotEncoder(categorical_features=[3])
    X = onehotencoder.fit_transform(X).toarray()
    X = X[: , 1:]
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state=0)

    #Step 2: Fitting Multiple Linear Regression to the Training set
    regressor = LinearRegression()
    regressor = regressor.fit(X_train,Y_train)
    #Step 3: Predicting the Test set results
    y_pred = regressor.predict(X_test)
    print(y_pred)

if __name__ == "__main__":
    multiple_linear_regression()