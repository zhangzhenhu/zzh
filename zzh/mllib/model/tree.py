#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2014 Hal.com, Inc. All Rights Reserved
#
"""
模块用途描述

Authors: lixiaolong(lixiaolong1)
Date:    2019/4/1 14:55
"""
import xgboost as xgb
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier, XGBRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
import os
from zzh.mllib.model import ABCModel

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class Xgboost(ABCModel):
    name = "xgboost"
    # 模型参数
    default = {'max_depth': 4,
               'min_child_weight': 4,
               'booster': 'gbtree',
               'n_estimators': 500,
               'learning_rate': 0.02,
               'gamma': 0.2,
               'silent': 1,
               'objective': 'binary:logistic',
               'subsample': 0.8,
               'colsample_bytree': 0.8}
    # min_child_weight:, max_depth:, gamma:,
    description = "xgboost"

    def fit(self, x, y, **options):
        # self.param = param
        # params = {'max_depth': range(2, 6, 1)}
        # params = {'min_child_weight': range(1, 6, 1)}
        # params = {'gamma': [0, 0.1, 0.2, 0.3]}
        # self.adjust_params(params)

        #  self.train_x = self.select_features(self.train_x)
        self.m = XGBClassifier(**self.model_params)
        print('model XGB fit begin:')
        self.m.fit(x, y, **options)

        # 评估训练集上的效果
        # print('model XGB predict begin:')
        # self.train_y_pred = self.predict(self.train_x)
        # self.train_y = np.array(self.train_y)
        # self.train_y_pred = np.array(self.train_y_pred)

        return self

    # def model_importiance_feature(self):
    #     feat_imp = pd.Series(self.model.feature_importances_, self.feature).sort_values(ascending=False)
    #     plt.figure(figsize=(20, 6))
    #     feat_imp.plot(kind='bar', title='model %s Feature Importances' % self.name)
    #     plt.ylabel('Feature Importance Score')
    #     plt.show()
    #
    # def train_result_evaluate(self, threshold=0.5):
    #     self.train_ev = self.evaluation.evaluate(y_true=self.train_y, y_pred=self.train_y_pred, threshold=threshold)
    #
    # def train_confusion_matrix(self, threshold=0.5):
    #     self.evaluation.confusion_matrix(y_true=self.train_y, y_pred=self.train_y_pred, threshold=threshold)
    #
    # def adjust_params(self, params=None, CV=5, scoring='roc_auc', n_iter=10):
    #     self.cvmodel = GridSearchCV(self.model, params, cv=CV, scoring=scoring, n_jobs=os.cpu_count())
    #     self.cvmodel.fit(self.train_x, self.train_y)
    #     self.best_params = self.cvmodel.best_params_
    #     self.best_score_ = self.cvmodel.best_score_
    #     print('%s在%s参数下，训练集准确率最高为%s:' % (self.name, self.best_params, self.best_score_))
    #     # 赋值模型以最优的参数
    #     # self.model = XGBClassifier(**self.param, max_depth=self.best_params['max_depth'])
    #     # self.model = XGBClassifier(**self.param, min_child_weight=self.best_params['min_child_weight'])
    #     self.model = XGBClassifier(**self.param)
    #     return self.model

    def predict(self, x):
        assert x is not None

        y_prob = self.model.predict_proba(x)
        y_prob = y_prob.tolist()
        y_prob = [item[1] for item in y_prob]
        y_prob = np.array(y_prob)

        # self.y_pred = y_prob

        return y_prob


class GradientBoostingTree(ABCModel):
    name = "GBDT"
    # 模型参数
    #     param = {'n_estimators': 30, 'learning_rate': 0.1, 'max_depth': 4, 'min_samples_split': 200,
    #              'min_samples_leaf': 150, 'max_features': None, 'subsample': 0.8, 'random_state': 10}

    description = "GBDT"

    def fit(self, x, y, **options):
        # self.param = param
        self.m = GradientBoostingClassifier(**self.model_params)

        #         print('begin GridSearch')
        #         params={'min_samples_leaf':list(range(100,301,50)),'min_samples_split':list(range(700,901,50))}
        #         self.adjust_params(params)

        # print('model GBDT fit begin:')
        self.m.fit(x, y)

        # 评估训练集上的效果
        # print('model GBDT predict begin:')
        # self.train_y_pred = self.predict(self.train_x)
        # self.train_y = np.array(self.train_y)
        # self.train_y_pred = np.array(self.train_y_pred)

        return self

    def model_importiance_feature(self):
        feat_imp = pd.Series(self.model.feature_importances_, self.feature).sort_values(ascending=False)
        plt.figure(figsize=(20, 6))
        feat_imp.plot(kind='bar', title='model %s Feature Importances' % self.name)
        plt.ylabel('Feature Importance Score')
        plt.show()

    def train_result_evaluate(self, threshold=0.5):
        self.train_ev = self.evaluation.evaluate(y_true=self.train_y, y_pred=self.train_y_pred, threshold=threshold)

    def train_confusion_matrix(self, threshold=0.5):
        self.evaluation.confusion_matrix(y_true=self.train_y, y_pred=self.train_y_pred, threshold=threshold)

    def adjust_params(self, params=None, CV=5, scoring='roc_auc', n_iter=10):
        self.cvmodel = GridSearchCV(self.model, params, cv=CV, scoring=scoring, n_jobs=-1)
        self.cvmodel.fit(self.train_x, self.train_y)
        self.cv_result = pd.DataFrame.from_dict(self.cvmodel.cv_results_)
        self.best_params = self.cvmodel.best_params_
        self.best_score_ = self.cvmodel.best_score_
        print(self.cv_result)
        print('%s在%s参数下，训练集AUC最高为%s:' % (self.name, self.best_params, self.best_score_))

        return self.model

    def predict(self, x):
        assert x is not None

        y_prob = self.model.predict_proba(x)
        y_prob = y_prob.tolist()
        y_prob = [item[1] for item in y_prob]
        y_prob = np.array(y_prob)

        # self.y_pred = y_prob

        return y_prob


class GBDT_LR(ABCModel):
    name = "GBDT_LR"
    # 模型参数
    #     param = {'n_estimators': 150, 'learning_rate': 0.1, 'max_depth': 6, 'min_samples_split': 800,
    #              'min_samples_leaf': 150, 'max_features': None, 'subsample': 0.8, 'random_state': 10}

    description = "GBDT_LR"

    def fit(self, x, y, **options):
        # self.param = param
        # print('model GBDT_LR fit begin:')
        grd = GradientBoostingClassifier(**self.model_params)
        grd_enc = OneHotEncoder()
        grd_lm = LogisticRegression(penalty='l2', C=1, solver='lbfgs')
        grd.fit(x, y)
        grd_enc.fit(grd.apply(x)[:, :, 0])
        grd_lm.fit(grd_enc.transform(grd.apply(self.train_x)[:, :, 0]), self.train_y)
        self.grd = grd
        self.grd_enc = grd_enc
        self.m = grd_lm

        # 评估训练集上的效果
        # print('model GBDT_LR predict begin:')
        # self.train_y_pred = self.predict(self.train_x)
        # self.train_y = np.array(self.train_y)
        # self.train_y_pred = np.array(self.train_y_pred)

        return self

    def model_importiance_feature(self):
        pass

    def train_result_evaluate(self, threshold=0.5):
        self.train_ev = self.evaluation.evaluate(y_true=self.train_y, y_pred=self.train_y_pred, threshold=threshold)

    def train_confusion_matrix(self, threshold=0.5):
        self.evaluation.confusion_matrix(y_true=self.train_y, y_pred=self.train_y_pred, threshold=threshold)

    def predict(self, x):
        assert x is not None

        y_prob = self.model.predict_proba(self.grd_enc.transform(self.grd.apply(x)[:, :, 0]))
        y_prob = y_prob.tolist()
        y_prob = [item[1] for item in y_prob]
        y_prob = np.array(y_prob)

        return y_prob


class RF_LR(ABCModel):
    name = "RF_LR"
    # 模型参数
    #     param = {'n_estimators': 180, 'n_jobs': -1, 'max_depth': 8, 'min_samples_split': 80,
    #              'min_samples_leaf': 70, 'max_features': None}

    description = "RF_LR"

    def fit(self, feature_list, threshold=0.5, **param):
        self.param = param
        rf = RandomForestClassifier(**self.param)
        rf_enc = OneHotEncoder()
        rf_lm = LogisticRegression(penalty='l2', C=1, solver='lbfgs')
        rf.fit(self.train_x, self.train_y)
        rf_enc.fit(rf.apply(self.train_x))
        rf_lm.fit(rf_enc.transform(rf.apply(self.train_x)), self.train_y)
        self.rf = rf
        self.rf_enc = rf_enc
        self.model = rf_lm

        # 评估训练集上的效果
        self.train_y_pred = self.predict(self.train_x)
        self.train_y = np.array(self.train_y)
        self.train_y_pred = np.array(self.train_y_pred)

        return self

    def model_importiance_feature(self):
        pass

    def train_result_evaluate(self, threshold=0.5):
        self.train_ev = self.evaluation.evaluate(y_true=self.train_y, y_pred=self.train_y_pred, threshold=threshold)

    def train_confusion_matrix(self, threshold=0.5):
        self.evaluation.confusion_matrix(y_true=self.train_y, y_pred=self.train_y_pred, threshold=threshold)

    def predict(self, x):
        assert x is not None

        y_prob = self.model.predict_proba(self.rf_enc.transform(self.rf.apply(x)))
        y_prob = y_prob.tolist()
        y_prob = [item[1] for item in y_prob]
        y_prob = np.array(y_prob)

        # self.y_pred = y_prob

        return y_prob


class RandomForest(ABCModel):
    name = "random_forest"

    #     param = {'n_estimators': 180, 'n_jobs': -1, 'max_depth': 8, 'min_samples_split': 140,
    #              'min_samples_leaf': 70, 'max_features': None}

    description = "RF"

    def fit(self, **param):
        self.param = param
        self.model = RandomForestClassifier(**self.param)
        #         params = {'max_depth': range(5, 15, 2),'min_samples_split':list(range(80,131,10))}
        #         params = {'n_estimators': range(100, 601, 50)}
        #         params = {'max_features': range(5, 28, 1)}
        #         params = {'min_samples_split': list(range(100, 121, 10)),'min_samples_leaf':list(range(30,71,10))}
        #         self.adjust_params(params)

        print('model RF fit begin:')
        self.model.fit(self.train_x, self.train_y)

        # 评估训练集上的效果
        print('model RF predict begin:')
        self.train_y_pred = self.predict(self.train_x)
        self.train_y = np.array(self.train_y)
        self.train_y_pred = np.array(self.train_y_pred)

        return self

    def model_importiance_feature(self):
        feat_imp = pd.Series(self.model.feature_importances_, self.feature).sort_values(ascending=False)
        plt.figure(figsize=(20, 6))
        feat_imp.plot(kind='bar', title='model %s Feature Importances' % self.name)
        plt.ylabel('Feature Importance Score')
        plt.show()

    def train_result_evaluate(self, threshold=0.5):
        self.train_ev = self.evaluation.evaluate(y_true=self.train_y, y_pred=self.train_y_pred, threshold=threshold)

    def train_confusion_matrix(self, threshold=0.5):
        self.evaluation.confusion_matrix(y_true=self.train_y, y_pred=self.train_y_pred, threshold=threshold)

    def adjust_params(self, params=None, CV=5, scoring='f1', n_iter=10):
        self.cvmodel = GridSearchCV(self.model, params, cv=CV, scoring=scoring, n_jobs=-1)
        self.cvmodel.fit(self.train_x, self.train_y)
        self.cv_result = pd.DataFrame.from_dict(self.cvmodel.cv_results_)
        self.best_params = self.cvmodel.best_params_
        self.best_score_ = self.cvmodel.best_score_
        print(self.cv_result)
        print('%s在%s参数下，训练集f1最高为%s:' % (self.name, self.best_params, self.best_score_))

    def predict(self, x):
        assert x is not None

        y_prob = self.model.predict_proba(x)
        y_prob = y_prob.tolist()
        y_prob = [item[1] for item in y_prob]
        y_prob = np.array(y_prob)

        # self.y_pred = y_prob

        return y_prob


class DecisionTree(ABCModel):
    name = "DecisionTree"
    # 模型参数
    param = {'max_depth': 9, 'min_samples_split': 220, 'min_samples_leaf': 18, 'criterion': 'gini',
             'max_features': None}
    #
    description = "决策树"

    def fit(self):
        self.model = DecisionTreeClassifier(**self.param)
        # params = {'min_samples_leaf': range(14, 26, 2)}
        # params = {'max_features': range(5, 44, 2)}
        # params = {'max_depth': range(4, 12, 1), 'min_samples_split': range(40, 260, 20)}
        # params = {'criterion': ['gini', 'entropy']}
        # self.adjust_params(params)
        self.model.fit(self.train_x, self.train_y)
        # 评估训练集上的效果
        self.train_y_pred = self.predict(self.train_x)

        return self

    def model_importiance_feature(self):
        feat_imp = pd.Series(self.model.feature_importances_, self.feature_list).sort_values(ascending=False)
        plt.figure(figsize=(20, 6))
        feat_imp.plot(kind='bar', title='model %s Feature Importances' % self.name)
        plt.ylabel('Feature Importance Score')
        plt.show()

    def train_result_evaluate(self, threshold=0.5):
        self.train_ev = self.evaluation.evaluate(y_true=self.train_y, y_pred=self.train_y_pred, threshold=threshold)

    def train_confusion_matrix(self, threshold=0.5):
        self.evaluation.confusion_matrix(y_true=self.train_y, y_pred=self.train_y_pred, threshold=threshold)

    def adjust_params(self, params=None, CV=10, scoring='accuracy', n_iter=10):
        self.cvmodel = GridSearchCV(self.model, params, cv=CV, scoring=scoring, n_jobs=os.cpu_count())
        self.cvmodel.fit(self.train_x, self.train_y)
        self.best_params = self.cvmodel.best_params_
        self.best_score_ = self.cvmodel.best_score_
        print('%s在%s参数下，训练集准确率最高为%s:' % (self.name, self.best_params, self.best_score_))

        self.model = DecisionTreeClassifier(**self.param)
        # self.model.fit(self.train_x, self.train_y)
        return self.cvmodel

    def predict(self, x):
        assert x is not None

        y_prob = self.model.predict_proba(x)
        y_prob = y_prob.tolist()
        y_prob = [item[1] for item in y_prob]
        y_prob = np.array(y_prob)

        # self.y_pred = y_prob

        return y_prob
