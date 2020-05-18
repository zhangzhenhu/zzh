#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2014 Hal.com, Inc. All Rights Reserved
#
"""
模块用途描述

Authors: 张振虎
Date:    2019/4/14 14:39
"""

import pandas as pd
import numpy as np
import random
import argparse
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.externals import joblib
from zzh.mllib.model import ABCModel


# from zzh.mllib.evaluation import Evaluation


class LR(ABCModel):
    name = "LogisticRegression"
    # 模型参数
    default = {
        "solver": "lbfgs",
    }
    description = "逻辑回归"

    def fit(self, x, y, **options):
        # self.feature_list = kwargs.get('feature_list', None)
        # solver_name = kwargs.get('solver_name', 'lbfgs')
        # penalty = kwargs.get('penalty', 'l2')
        # k_single = kwargs.get('k_single', 0)
        # k_pca = kwargs.get('k_pca', 1)

        # self.train_x, self.train_y = self.tf_sample(self.train_x, self.train_y)

        # 数据归一化
        # scaler = preprocessing.StandardScaler()
        # self.scalar_ = scaler.fit(self.train_x)

        # pca
        # selection = SelectKBest(k=k_single)
        # n_components = int(len(self.feature_names) * k_pca)
        # pca = PCA(n_components=n_components)
        # pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
        # combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])
        # self.pca = combined_features.fit(self.train_x, self.train_y)
        # self.pca = PCA(n_components=n_components).fit(self.train_x)

        self.m = LogisticRegression(**self.model_params)
        # self.m = LogisticRegression(penalty=penalty, C=1, solver=solver_name)
        # fit_data = self.train_x.copy()
        # fit_data = self.scalar_.transform(fit_data)
        # fit_data = self.pca.transform(fit_data)
        self.m.fit(x, y, **options)

        # 评估训练集上的效果
        # self.train_y_pred = self.predict(self.train_x)
        # self.train_y = np.array(self.train_y)
        # self.train_y_pred = np.array(self.train_y_pred)
        # self.train_ev = self.evaluation.evaluate(y_true=self.train_y, y_pred=self.train_y_pred, threshold=0.5)

        return self

    def predict_prob(self, x, **options):
        assert x
        # assert self.m

        # x = self.scalar_.transform(x)
        # x = self.pca.transform(x)

        y_prob = self.m.predict_proba(x)
        # y_prob = y_prob.tolist()
        # y_prob = [item[1] for item in y_prob]
        # y_prob = np.array(y_prob)

        return y_prob

    def predict(self, x, thereshold=0.5):
        y_prob = self.predict_prob(x)
        y_prob[y_prob >= 0.5] = 1
        y_prob[y_prob < 0.5] = 1
        return y_prob

    def load(self, model_file):
        pass
        self.m = joblib.load(model_file)

    def save(self, save_path):
        joblib.dump(self.m, save_path)
