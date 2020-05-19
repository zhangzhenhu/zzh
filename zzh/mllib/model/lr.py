#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
#
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
# from sklearn.model_selection import GridSearchCV
# from sklearn.decomposition import PCA, KernelPCA
# from sklearn.feature_selection import SelectKBest
# from sklearn.pipeline import Pipeline, FeatureUnion
# from sklearn.externals import joblib
from zzh.mllib.model import ABCModel
from zzh.mllib.feature import DataSet


# from zzh.mllib.evaluation import Evaluation


class LR(ABCModel):
    name = "LogisticRegression"
    # 模型参数
    default = {
        "solver": "lbfgs",
    }
    description = "逻辑回归"

    def _fit(self, dataset: DataSet, **options):
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
        self.m.fit(dataset.x, dataset.y, **options)

