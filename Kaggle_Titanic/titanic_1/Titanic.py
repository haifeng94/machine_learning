# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold
from sklearn import model_selection

'''
Titanic solution 1:主要根据数值型数据进行简单的数据处理，然后利用线性回归及逻辑回归进行简单预测
'''

# 线性回归
def linearRegress(data_train):
    # 数值处理，Age列缺失值使用中位数填充
    data_train = data_train
    data_train['Age'] = data_train['Age'].fillna(data_train['Age'].median())

    # 提取简单的特征(主要提取了数值型特征)
    predictors = ["Pclass","Age","SibSp","Parch","Fare"]

    # 初始化算法
    lr = LinearRegression()

    # 样本平均分成3份，3折交叉验证
    kf = KFold(n_splits=3, shuffle=False, random_state=1)

    predictions = []
    for train,test in kf.split(data_train):
        train_predictors = data_train[predictors].iloc[train,:]
        train_target = data_train['Survived'].iloc[train]
        lr.fit(train_predictors,train_target)

        test_predictions = lr.predict(data_train[predictors].iloc[test,:])
        predictions.append(test_predictions)

    # array聚合
    predictions = np.concatenate(predictions,axis=0)

    predictions[predictions > .5] = 1
    predictions[predictions <= .5] = 0

    accuracy = sum(predictions == data_train["Survived"]) / len(predictions)
    print("准确率为: ", accuracy)

# 逻辑回归
def logRegress(data_train):
    # 数值处理，Age列缺失值使用中位数填充
    data_train = data_train
    data_train['Age'] = data_train['Age'].fillna(data_train['Age'].median())

    # 提取简单的特征(主要提取了数值型特征)
    predictors = ["Pclass", "Age", "SibSp", "Parch", "Fare"]

    # 初始化逻辑回归
    logReg = LogisticRegression(random_state=1)

    logReg.fit(data_train[predictors], data_train['Survived'])

    # 使用sklearn库里面的交叉验证函数获取预测准确率分数
    scores = model_selection.cross_val_score(logReg,data_train[predictors],data_train['Survived'],cv=3)

    # 使用交叉验证分数的平均值作为最终的准确率
    print("准确率为: ", scores.mean())

if __name__ == "__main__":
    data_train = pd.read_csv('train.csv')
    data_test = pd.read_csv('test.csv')

    # 线性回归简单处理
    #linearRegress(data_train)

    # 逻辑回归简单处理
    logRegress(data_train)
