# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import re
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn import ensemble
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

def data_process():
    # 读取文件
    data_train = pd.read_csv('train.csv') # 891rows
    data_test = pd.read_csv('test.csv')   # 418rows

    # 合并训练集和测试集
    data_test['Survived'] = -1
    PassengerId = data_test['PassengerId']
    data_train_test = data_train.append(data_test) # 891+418=1309rows, 12columns

    # 数据处理
    # Embarked
    # Embarked使用众数来填充
    data_train_test['Embarked'].fillna(data_train_test['Embarked'].mode().iloc[0], inplace=True)
    # Embarked,数值转换，我们知道可以有两种特征处理方式；dummy和factorizing
    data_train_test['Embarked'] = pd.factorize(data_train_test['Embarked'])[0]
    # 使用pd.get_dummies获取one-hot编码
    emb_dummies = pd.get_dummies(data_train_test['Embarked'], prefix=data_train_test[['Embarked']].columns[0])
    data_train_test = pd.concat([data_train_test, emb_dummies], axis=1)

    # Sex
    # Sex特征进行factorizing
    data_train_test['Sex'] = pd.factorize(data_train_test['Sex'])[0]
    # 使用pd.get_dummies获取one-hot编码
    sex_dummies = pd.get_dummies(data_train_test['Sex'], prefix=data_train_test[['Sex']].columns[0])
    data_train_test = pd.concat([data_train_test, sex_dummies], axis=1)

    # Name
    # 提取名字中的各种称呼
    data_train_test['Title'] = data_train_test['Name'].map(lambda x: re.compile(",(.*?)\.").findall(x)[0])
    data_train_test['Title'] = data_train_test['Title'].apply(lambda x: x.strip())
    # 对各种称呼进行统一化处理
    title_Dict = {}
    title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
    title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
    title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
    title_Dict.update(dict.fromkeys(['Male', 'Miss'], 'Miss'))
    title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
    title_Dict.update(dict.fromkeys(['Master', 'Jonkheer'], 'Master'))
    data_train_test['Title'] = data_train_test['Title'].map(title_Dict)
    # 将Title特征进行factorizing
    data_train_test['Title'] = pd.factorize(data_train_test['Title'])[0]
    # 使用pd.get_dummies获取one-hot编码
    title_dummies = pd.get_dummies(data_train_test['Title'], prefix=data_train_test[['Title']].columns[0])
    data_train_test = pd.concat([data_train_test,title_dummies], axis=1)
    # 增加名字长度的特征
    data_train_test['Name_length'] = data_train_test['Name'].apply(len)

    # Fare
    # 缺失值处理，均值mean(),transform将函数np.mean应用到各个group中
    data_train_test['Fare'] = data_train_test[['Fare']].fillna(data_train_test.groupby('Pclass').transform(np.mean))
    # 将家庭票和团体票平均分配到个人
    data_train_test['Group_Ticket'] = data_train_test['Fare'].groupby(by=data_train_test['Ticket']).transform('count')
    data_train_test['Fare'] = data_train_test['Fare'] / data_train_test['Group_Ticket']
    data_train_test.drop(['Group_Ticket'], axis=1, inplace=True)
    # 使用binning给票价分等级
    data_train_test['Fare_bin'] = pd.qcut(data_train_test['Fare'], 5)
    # 进行factorizing
    data_train_test['Fare_bin_id'] = pd.factorize(data_train_test['Fare_bin'])[0]
    fare_bin_dummies = pd.get_dummies(data_train_test['Fare_bin_id']).rename(columns = lambda x: 'Fare_' + str(x))
    data_train_test = pd.concat([data_train_test,fare_bin_dummies], axis=1)
    data_train_test.drop(['Fare_bin'], axis=1, inplace=True)

    # Pclass
    # 不同等级的船舱之外再设高价位及低价位
    Pclass1_mean_fare = data_train_test['Fare'].groupby(by=data_train_test['Pclass']).mean().get(1)
    Pclass2_mean_fare = data_train_test['Fare'].groupby(by=data_train_test['Pclass']).mean().get(2)
    Pclass3_mean_fare = data_train_test['Fare'].groupby(by=data_train_test['Pclass']).mean().get(3)
    data_train_test['Pclass_Fare_Category'] = data_train_test.apply(pclass_fare_category, args=(
        Pclass1_mean_fare, Pclass2_mean_fare, Pclass3_mean_fare), axis=1)
    pclass_level = LabelEncoder()
    # 给每一项添加标签
    pclass_level.fit(np.array(['Pclass1_Low','Pclass1_High','Pclass2_Low','Pclass2_High','Pclass3_Low','Pclass3_High']))
    # 转换成数值
    data_train_test['Pclass_Fare_Category'] = pclass_level.transform(data_train_test['Pclass_Fare_Category'])
    # dummy 转换
    pclass_dummies = pd.get_dummies(data_train_test['Pclass_Fare_Category']).rename(
        columns=lambda x: 'Pclass_' + str(x))
    data_train_test = pd.concat([data_train_test, pclass_dummies], axis=1)
    # 将Pclass特征factorize化
    data_train_test['Pclass'] = pd.factorize(data_train_test['Pclass'])[0]

    # Parch and SibSp
    data_train_test['Family_Size'] = data_train_test['Parch'] + data_train_test['SibSp'] + 1
    data_train_test['Family_Size_Category'] = data_train_test['Family_Size'].map(family_size_category)
    le_family = LabelEncoder()
    le_family.fit(np.array(['Single', 'Small_Family', 'Large_Family']))
    data_train_test['Family_Size_Category'] = le_family.transform(data_train_test['Family_Size_Category'])
    family_size_dummies = pd.get_dummies(data_train_test['Family_Size_Category'],
                                            prefix=data_train_test[['Family_Size_Category']].columns[0])
    data_train_test = pd.concat([data_train_test, family_size_dummies], axis=1)

    # Age
    # 结合几项如Sex、Title、Pclass等其他没有缺失值的项，使用机器学习算法来预测Age
    missing_age = pd.DataFrame(data_train_test[['Age', 'Embarked', 'Sex', 'Title', 'Name_length', 'Family_Size',
                    'Family_Size_Category', 'Fare', 'Fare_bin_id', 'Pclass']])

    age_train = missing_age[missing_age['Age'].notnull()]
    age_test = missing_age[missing_age['Age'].isnull()]
    data_train_test.loc[(data_train_test.Age.isnull()), 'Age'] = fill_missing_age(age_train,age_test)

    # Ticket
    # 将Ticket按字母数字进行分类
    data_train_test['Ticket_Letter'] = data_train_test['Ticket'].str.split().str[0]
    data_train_test['Ticket_Letter'] = data_train_test['Ticket_Letter'].apply(
        lambda x: 'U0' if x.isnumeric() else x)
    # 将 Ticket_Letter factorize
    data_train_test['Ticket_Letter'] = pd.factorize(data_train_test['Ticket_Letter'])[0]

    # Cabin
    # 根据有与无进行分类
    data_train_test.loc[data_train_test.Cabin.isnull(), 'Cabin'] = 'U0'
    data_train_test['Cabin'] = data_train_test['Cabin'].apply(lambda x: 0 if x == 'U0' else 1)

    # 正则化处理，(X-mean)/std
    name_age_fare = preprocessing.StandardScaler().fit(data_train_test[['Age', 'Fare', 'Name_length']])
    data_train_test[['Age', 'Fare', 'Name_length']] = name_age_fare.transform(
        data_train_test[['Age', 'Fare', 'Name_length']])

    # 弃掉无用特征
    data_backup = data_train_test

    data_train_test.drop(['PassengerId', 'Embarked', 'Sex', 'Name', 'Fare_bin_id', 'Pclass_Fare_Category',
                              'Parch', 'SibSp', 'Family_Size_Category', 'Ticket'], axis=1, inplace=True)

    # 将训练集和测试集分开
    train_data = data_train_test[:891]
    test_data = data_train_test[891:]

    train_data_X = train_data.drop(['Survived'], axis=1)
    train_data_Y = train_data['Survived']
    test_data_X = test_data.drop(['Survived'], axis=1)
    print(data_train_test)

    return train_data_X, train_data_Y, test_data_X,PassengerId

def fill_missing_age(age_train, age_test):
    age_X_train = age_train.drop(['Age'], axis=1)
    age_Y_train = age_train['Age']
    age_X_test = age_test.drop(['Age'], axis=1)

    # model 1  gbm
    gbm_reg = GradientBoostingRegressor(random_state=42)
    gbm_reg_param_grid = {'n_estimators': [2000], 'max_depth': [4], 'learning_rate': [0.01], 'max_features': [3]}
    gbm_reg_grid = model_selection.GridSearchCV(gbm_reg, gbm_reg_param_grid, cv=10, n_jobs=25, verbose=1,
                                                scoring='neg_mean_squared_error')
    gbm_reg_grid.fit(age_X_train, age_Y_train)
    print('Age feature Best GB Params:' + str(gbm_reg_grid.best_params_))
    print('Age feature Best GB Score:' + str(gbm_reg_grid.best_score_))
    print('GB Train Error for "Age" Feature Regressor:' + str(
        gbm_reg_grid.score(age_X_train, age_Y_train)))
    age_test.loc[:, 'Age_GB'] = gbm_reg_grid.predict(age_X_test)
    print(age_test['Age_GB'][:4])

    # model 2 rf
    rf_reg = RandomForestRegressor()
    rf_reg_param_grid = {'n_estimators': [200], 'max_depth': [5], 'random_state': [0]}
    rf_reg_grid = model_selection.GridSearchCV(rf_reg, rf_reg_param_grid, cv=10, n_jobs=25, verbose=1,
                                               scoring='neg_mean_squared_error')
    rf_reg_grid.fit(age_X_train, age_Y_train)
    print('Age feature Best RF Params:' + str(rf_reg_grid.best_params_))
    print('Age feature Best RF Score:' + str(rf_reg_grid.best_score_))
    print(
        'RF Train Error for "Age" Feature Regressor' + str(rf_reg_grid.score(age_X_train, age_Y_train)))
    age_test.loc[:, 'Age_RF'] = rf_reg_grid.predict(age_X_test)
    print(age_test['Age_RF'][:4])

    # two models merge
    print('shape1', age_test['Age'].shape, age_test[['Age_GB', 'Age_RF']].mode(axis=1).shape)
    # missing_age_test['Age'] = missing_age_test[['Age_GB', 'Age_LR']].mode(axis=1)

    age_test.loc[:, 'Age'] = np.mean([age_test['Age_GB'], age_test['Age_RF']])
    print(age_test['Age'][:4])

    age_test.drop(['Age_GB', 'Age_RF'], axis=1, inplace=True)

    return age_test

# 建立Pclass Fare Category
def pclass_fare_category(df,pclass1_mean_fare,pclass2_mean_fare,pclass3_mean_fare):
    if df['Pclass'] == 1:
        if df['Fare'] <= pclass1_mean_fare:
            return 'Pclass1_Low'
        else:
            return 'Pclass1_High'
    elif df['Pclass'] == 2:
        if df['Fare'] <= pclass2_mean_fare:
            return 'Pclass2_Low'
        else:
            return 'Pclass2_High'
    elif df['Pclass'] == 3:
        if df['Fare'] <= pclass3_mean_fare:
            return 'Pclass3_Low'
        else:
            return 'Pclass3_High'

def family_size_category(family_size):
    if family_size <= 1:
        return 'Single'
    elif family_size <= 4:
        return 'Small_Family'
    else:
        return 'Large_Family'

if __name__ == "__main__":
    data_process()