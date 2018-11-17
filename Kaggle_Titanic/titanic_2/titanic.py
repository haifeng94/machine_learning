# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import dataProcess
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


def get_top_n_features(train_data_X,train_data_Y,top_n_features):
    # randomforest
    rf = RandomForestClassifier(random_state=0)
    rf_param = {'n_estimators':[500],'min_samples_split':[2,3],'max_depth':[20]}
    rf_grid = model_selection.GridSearchCV(rf, rf_param, n_jobs=25, cv=10, verbose=1)
    rf_grid.fit(train_data_X,train_data_Y)
    print('Top N Features Best RF Params:' + str(rf_grid.best_params_))
    print('Top N Features Best RF Score:' + str(rf_grid.best_score_))
    print('Top N Features RF Train Score:' + str(rf_grid.score(train_data_X, train_data_Y)))
    feature_imp_sorted_rf = pd.DataFrame({'feature':list(train_data_X),
            'importance':rf_grid.best_estimator_.feature_importances_}).sort_values('importance',ascending=False)
    features_top_n_rf = feature_imp_sorted_rf.head(top_n_features)['feature']
    print('Sample 10 Feeatures from RF Classifier')
    print(str(features_top_n_rf[:10]))

    # AdaBoost
    ada_est = AdaBoostClassifier(random_state=0)
    ada_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1]}
    ada_grid = model_selection.GridSearchCV(ada_est, ada_param_grid, n_jobs=25, cv=10, verbose=1)
    ada_grid.fit(train_data_X, train_data_Y)
    print('Top N Features Best Ada Params:' + str(ada_grid.best_params_))
    print('Top N Features Best Ada Score:' + str(ada_grid.best_score_))
    print('Top N Features Ada Train Score:' + str(ada_grid.score(train_data_X, train_data_Y)))
    feature_imp_sorted_ada = pd.DataFrame({'feature': list(train_data_X),
            'importance': ada_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_ada = feature_imp_sorted_ada.head(top_n_features)['feature']
    print('Sample 10 Features from Ada Classifier:')
    print(str(features_top_n_ada[:10]))

    # ExtraTree
    et_est = ExtraTreesClassifier(random_state=0)
    et_param_grid = {'n_estimators': [500], 'min_samples_split': [3, 4], 'max_depth': [20]}
    et_grid = model_selection.GridSearchCV(et_est, et_param_grid, n_jobs=25, cv=10, verbose=1)
    et_grid.fit(train_data_X, train_data_Y)
    print('Top N Features Best ET Params:' + str(et_grid.best_params_))
    print('Top N Features Best DT Score:' + str(et_grid.best_score_))
    print('Top N Features ET Train Score:' + str(et_grid.score(train_data_X, train_data_Y)))
    feature_imp_sorted_et = pd.DataFrame({'feature': list(train_data_X),
        'importance': et_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_et = feature_imp_sorted_et.head(top_n_features)['feature']
    print('Sample 10 Features from ET Classifier:')
    print(str(features_top_n_et[:10]))

    # GradientBoosting
    gb_est = GradientBoostingClassifier(random_state=0)
    gb_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1], 'max_depth': [20]}
    gb_grid = model_selection.GridSearchCV(gb_est, gb_param_grid, n_jobs=25, cv=10, verbose=1)
    gb_grid.fit(train_data_X, train_data_Y)
    print('Top N Features Best GB Params:' + str(gb_grid.best_params_))
    print('Top N Features Best GB Score:' + str(gb_grid.best_score_))
    print('Top N Features GB Train Score:' + str(gb_grid.score(train_data_X, train_data_Y)))
    feature_imp_sorted_gb = pd.DataFrame({'feature': list(train_data_X),
            'importance': gb_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_gb = feature_imp_sorted_gb.head(top_n_features)['feature']
    print('Sample 10 Feature from GB Classifier:')
    print(str(features_top_n_gb[:10]))

    # DecisionTree
    dt_est = DecisionTreeClassifier(random_state=0)
    dt_param_grid = {'min_samples_split': [2, 4], 'max_depth': [20]}
    dt_grid = model_selection.GridSearchCV(dt_est, dt_param_grid, n_jobs=25, cv=10, verbose=1)
    dt_grid.fit(train_data_X, train_data_Y)
    print('Top N Features Bset DT Params:' + str(dt_grid.best_params_))
    print('Top N Features Best DT Score:' + str(dt_grid.best_score_))
    print('Top N Features DT Train Score:' + str(dt_grid.score(train_data_X, train_data_Y)))
    feature_imp_sorted_dt = pd.DataFrame({'feature': list(train_data_X),
        'importance': dt_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_dt = feature_imp_sorted_dt.head(top_n_features)['feature']
    print('Sample 10 Features from DT Classifier:')
    print(str(features_top_n_dt[:10]))

    # 模型融合
    features_top_n = pd.concat([features_top_n_rf, features_top_n_ada, features_top_n_et, features_top_n_gb,
                                features_top_n_dt],ignore_index=True).drop_duplicates()

    features_importance = pd.concat([feature_imp_sorted_rf, feature_imp_sorted_ada, feature_imp_sorted_et,
                                 feature_imp_sorted_gb, feature_imp_sorted_dt], ignore_index=True)

    return features_top_n, features_importance

def get_out_fold(clf,x_train,y_train,x_test,ntrain,ntest):
    kf = KFold(n_splits=7, random_state=0, shuffle=False)
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((7,ntest))

    for i, (train_index,test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.fit(x_tr,y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i,:] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1) # reshape(-1, 1)表示转成一列

if __name__ == "__main__":
    train_data_X, train_data_Y, test_data_X, PassengerId = dataProcess.data_process()
    feature_to_pick = 30
    feature_top_n, feature_importance = get_top_n_features(train_data_X, train_data_Y, feature_to_pick)
    train_data_X = pd.DataFrame(train_data_X[feature_top_n])
    test_data_X = pd.DataFrame(test_data_X[feature_top_n])

    ntrain = train_data_X.shape[0]
    ntest = test_data_X.shape[0]

    # 将pandas转换为arrays
    x_train = train_data_X.values
    y_train = train_data_Y.values
    x_test = test_data_X.values

    # Level1对每个基学习器使用K-fold，将Kge模型对Valid Set的预测结果拼起来，作为下一层学习器的输入
    # Random Forest
    rf = RandomForestClassifier(n_estimators=500, warm_start=True, max_features='sqrt', max_depth=6,
                                min_samples_split=3, min_samples_leaf=2, n_jobs=-1, verbose=0)
    rf_oof_train, rf_oof_test = get_out_fold(rf, x_train, y_train, x_test, ntrain, ntest)
    # AdaBoost
    ada = AdaBoostClassifier(n_estimators=500, learning_rate=0.1)
    ada_oof_train, ada_oof_test = get_out_fold(ada, x_train, y_train, x_test, ntrain, ntest)
    # Extra Trees
    et = ExtraTreesClassifier(n_estimators=500, n_jobs=-1, max_depth=8, min_samples_leaf=2, verbose=0)
    et_oof_train, et_oof_test = get_out_fold(et, x_train, y_train, x_test, ntrain, ntest)
    # Gradient Boost
    gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.008, min_samples_split=3, min_samples_leaf=2,
                                    max_depth=5, verbose=0)
    gb_oof_train, gb_oof_test = get_out_fold(gb, x_train, y_train, x_test, ntrain, ntest)
    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=8)
    dt_oof_train, dt_oof_test = get_out_fold(dt, x_train, y_train, x_test, ntrain, ntest)
    # KNeighbors
    knn = KNeighborsClassifier(n_neighbors=2)
    knn_oof_train, knn_oof_test = get_out_fold(knn, x_train, y_train, x_test, ntrain, ntest)
    # Support Vector
    svm = SVC(kernel='linear', C=0.025)
    svm_oof_train, svm_oof_test = get_out_fold(svm, x_train, y_train, x_test, ntrain, ntest)

    x_train = np.concatenate(
        (rf_oof_train, ada_oof_train, et_oof_train, gb_oof_train, dt_oof_train, knn_oof_train, svm_oof_train), axis=1)
    x_test = np.concatenate(
        (rf_oof_test, ada_oof_test, et_oof_test, gb_oof_test, dt_oof_test, knn_oof_test, svm_oof_test), axis=1)

    # Level 2： 利用XGBoost，使用第一层预测的结果作为特征对最终的结果进行预测。
    gbm = XGBClassifier(n_estimators=200, max_depth=4, min_child_weight=2, gamma=0.9, subsample=0.8,
        colsample_bytree=0.8, objective='binary:logistic', nthread=-1, scale_pos_weight=1).fit(x_train, y_train)
    predictions = gbm.predict(x_test)

    # 生成结果提交
    StackingSubmission = pd.DataFrame({'PassengerId': PassengerId, 'Survived': predictions})
    StackingSubmission.to_csv('StackingSubmission.csv', index=False, sep=',')