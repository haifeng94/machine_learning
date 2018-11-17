# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

'''
Titanic solution 1:主要在前面的数值型数据的基础上增加两个特征，然后进行简单的数据处理，最后利用随机森林及逻辑回归进行简单预测
'''

# 数据处理
def dataProcess(data):
    # Age列中的缺失值用Age中位数进行填充
    data['Age'] = data['Age'].fillna(data['Age'].median())

    # Fare列中的缺失值用Fare最大值进行填充
    data['Fare'] = data['Fare'].fillna(data['Fare'].max())

    # Embarked缺失值用最多的S进行填充
    data['Embarked'] = data['Embarked'].fillna('S')
    # Embarked用0,1,2代替S,C,Q
    data.loc[data['Embarked'] == 'S', 'Embarked'] = 0
    data.loc[data['Embarked'] == 'C', 'Embarked'] = 1
    data.loc[data['Embarked'] == 'Q', 'Embarked'] = 2

    # Sex性别列处理：male用0，female用1
    data.loc[data['Sex'] == 'male', 'Sex'] = 0
    data.loc[data['Sex'] == 'female','Sex'] =1

    return data

# 逻辑回归
def logRegress(train,test):
    data_train = dataProcess(train)
    data_test = dataProcess(test)
    predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

    logReg = LogisticRegression(random_state=1)
    logReg.fit(data_train[predictors], data_train['Survived'])

    scores = model_selection.cross_val_score(logReg,data_train[predictors],data_train['Survived'],cv=3)
    print("准确率为: ", scores.mean())

    # 构造测试集的Survived列
    data_test['Survived'] = -1

    data_test['Survived'] = logReg.predict(data_test[predictors])
    print(data_test['Survived'].head(10))

    # 保存为csv文档
    result = pd.DataFrame({'PassengerId': data_test['PassengerId'], 'Survived': data_test['Survived']})
    result.to_csv('result.csv', index=False, sep=',')

def randomForest(train,test):
    data_train = dataProcess(train)
    data_test = dataProcess(test)

    predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

    # 30棵决策树，停止的条件：样本个数为2，叶子节点个数为1
    rf = RandomForestClassifier(random_state=1,n_estimators=30,min_samples_split=2,min_samples_leaf=1)

    # 样本平均分成10份，10折交叉验证
    kf = KFold(n_splits=10,shuffle=False,random_state=1)

    rf.fit(data_train[predictors],data_train['Survived'])

    scores = model_selection.cross_val_score(rf,data_train[predictors],data_train['Survived'],cv=kf)
    print(scores)
    print(scores.mean())

    # 构造测试集的Survived列
    data_test['Survived'] = -1

    data_test['Survived'] = rf.predict(data_test[predictors])
    print(data_test['Survived'].head())

    # 保存为csv文档
    result = pd.DataFrame({'PassengerId': data_test['PassengerId'], 'Survived': data_test['Survived']})
    result.to_csv('result.csv', index=False, sep=',')

if __name__ == "__main__":
    data_train = pd.read_csv('train.csv')
    data_test = pd.read_csv('test.csv')

    # 逻辑回归
    #logRegress(data_train,data_test)

    # 随机森林
    randomForest(data_train,data_test)
