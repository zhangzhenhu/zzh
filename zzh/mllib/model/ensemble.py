#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2014 Hal.com, Inc. All Rights Reserved
#
"""
模块用途描述

Authors: zhangzhenhu
Date:    2019/4/4 14:39
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from .model import Model, Evaluation


class AdaBoost(Model):
    name = "AdaBoost"
    param = {'n_estimators': 14, 'learning_rate': 0.2}
    # 'n_estimators':, 'learning_rate':
    description = "AdaBoost"

    def fit(self):
        param_base = {'max_depth': 6, 'min_samples_split': 160, 'min_samples_leaf': 30, 'criterion': 'entropy',
                      'max_features': None}
        self.base_estimator = DecisionTreeClassifier(**param_base)
        self.model = AdaBoostClassifier(base_estimator=self.base_estimator)

        # params = {'min_samples_leaf': range(10, 60, 10)}
        # params = {'max_features': range(5, 44, 2)}
        # params = {'max_depth': range(4, 12, 1), 'min_samples_split': range(40, 260, 20)}
        # params = {'criterion': ['gini', 'entropy']}
        # params = {'n_estimators': range(5, 20, 1), 'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5]}
        # self.adjust_params(params)
        self.model.fit(self.train_x, self.train_y)
        # 评估训练集上的效果
        self.train_y_pred = self.predict(self.train_x)
        self.train_ev = self.evaluation.evaluate(y_true=self.train_y, y_pred=self.train_y_pred, threshold=0.5)
        return self

    def adjust_params(self, params=None, CV=10, scoring='accuracy', n_iter=10):
        self.cvmodel = GridSearchCV(self.model, params, cv=CV, scoring=scoring, n_jobs=-1)
        self.cvmodel.fit(self.train_x, self.train_y)
        self.best_params = self.cvmodel.best_params_
        self.best_score_ = self.cvmodel.best_score_
        print('%s在%s参数下，训练集准确率最高为%s:' % (self.name, self.best_params, self.best_score_))

        # 赋值模型以最优的参数
        self.model = AdaBoostClassifier(**self.param)
        self.model.fit(self.train_x, self.train_y)
        return self.model

    def predict(self, x):
        assert x is not None
        y_prob = self.model.predict_proba(x)
        y_prob = y_prob.tolist()
        y_prob = [item[1] for item in y_prob]
        y_prob = np.array(y_prob)
        return y_prob
