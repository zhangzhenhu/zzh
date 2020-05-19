#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
#
#
"""
模块用途描述

Authors: zhangzhenhu
Date:    2019/4/3 14:30
"""
from typing import Dict
import pandas as pd
import numpy as np
import random
import abc
import joblib
import matplotlib.pyplot as plt

from zzh.mllib.feature import DataSet
# from IPython.display import display
from zzh.mllib.evaluation import Evaluation


# from imblearn.over_sampling import SMOTE,ADASYN,RandomOverSampler
# from imblearn.under_sampling import RandomUnderSampler


class ABCModel(abc.ABC):
    # 模型名称
    name = "abstract"  # type: str
    # 模型参数K
    default = {}
    # 简单描述
    description = "模型的简单介绍"  # type: str
    # 使用的特征列表
    feature_names = []

    # feature = None
    # train_x = None  # type: pd.Dataframe
    # train_y = None  # type: np.ndarray
    # train_y_pred = None  # type: np.ndarray
    # test_x = None  # type: pd.Dataframe
    # test_y = None  # type: np.ndarray
    # test_y_pred = None  # type: np.ndarray
    # 评估结果
    # train_ev = None  # type: dict
    # test_ev = None  # type: dict

    def __init__(self, name=None, **options):
        # self.data_param = data_param
        self.m = None
        if name:
            self.name = name
        self.model_params = self.default.copy()
        if options:
            self.model_params.update(options)

        self.trainset = None
        self.testset = None

    def tf_sample(self, df_x, df_y):
        """
        调整正负样本比例
        :param df_x:
        :param df_y:
        :return:
        """
        se_1 = [i for i in range(len(df_y)) if int(df_y[i]) == 1]
        ratio = (1 - df_y.mean()) / df_y.mean()
        len_del = len(se_1) * (1 - ratio) * 0
        random.shuffle(se_1)
        se_1_sub = se_1[:int(len_del)]
        df_x = df_x[~df_x.index.isin(se_1_sub)]
        df_y = [df_y[i] for i in range(len(df_y)) if i not in se_1_sub]
        return df_x, df_y

    def fit(self, dataset: DataSet, **options):
        self.trainset = dataset.copy()
        self._fit(self.trainset, **options)
        self.trainset.predict = self.predict(dataset.x)
        return self

    @abc.abstractmethod
    def _fit(self, dataset: DataSet, **options):
        """

        :param dataset:
        :param options:
        :return:
        """
        raise NotImplemented

    def test(self, dataset, **options):
        self.testset = dataset
        self.testset.predict = self.predict(dataset.x)

    def predict(self, x, **options):

        y_prob = self.m.predict_proba(x)

        return y_prob

    def load(self, model_file):
        self.m = joblib.load(model_file)

    def save(self, save_path):
        joblib.dump(self.m, save_path)

    def evaluate(self, threshold=None):

        # dataset = DataSet().update(dataset)
        # dataset.predict = self.predict(dataset.x)
        if self.trainset and self.trainset.predict:
            train_ev = Evaluation(name=self.name, dataset=self.trainset).eval()
        else:
            train_ev = None
        if self.testset and self.testset.predict:
            test_ev = Evaluation(name=self.name, dataset=self.testset).eval()
        else:
            test_ev = None

        return train_ev, test_ev

    def importance_feature(self):
        feat_imp = pd.Series(self.m.feature_importances_, self.trainset.header).sort_values(ascending=False)
        plt.figure(figsize=(20, 6))
        feat_imp.plot(kind='bar', title='model %s Feature Importances' % self.name)
        plt.ylabel('Feature Importance Score')
        plt.show()

    # def test(self):
    #     """
    #     评估测试集上的效果
    #     :return:
    #     """
    #
    #     self.test_y_pred = self.predict(self.test_x)
    #
    # def test_result_evaluate(self, threshold=0.5):
    #     self.test_ev = self.evaluation.evaluate(y_true=self.test_y, y_pred=self.test_y_pred, threshold=threshold)
    #
    # def test_confusion_matrix(self, threshold=0.5):
    #     assert self.test_y is not None
    #     assert self.test_y_pred is not None
    #     self.evaluation.confusion_matrix(y_true=self.test_y, y_pred=self.test_y_pred, threshold=threshold)
