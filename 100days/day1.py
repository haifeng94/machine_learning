# coding=utf-8

import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Data process
def pre_process():
    #Importing dataset
    dataset = pd.read_csv('Data.csv')
    X = dataset.iloc[ : , :-1].values
    Y = dataset.iloc[ : , 3].values
    #print(X)
    #print(Y)
    #Handling the missing data
    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imputer = imputer.fit(X[ : , 1:3])
    X[ : , 1:3] = imputer.transform(X[ : , 1:3])
    #print(X)
    #Encoding categorical data
    labelencoder_X = LabelEncoder()
    labelencoder_Y = LabelEncoder()
    X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])
    Y = labelencoder_Y.fit_transform(Y)
    onehotencoder = OneHotEncoder(categorical_features = [0])
    X = onehotencoder.fit_transform(X).toarray()
    #print(X)
    #Splitting the datasets into training sets and Test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
    #print(X_test)
    #Feature Scaling
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.fit_transform(X_test)
    print(X_test)

if __name__ == "__main__":
    pre_process()