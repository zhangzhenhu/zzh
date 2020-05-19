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

    def __init__(self, evaluation=None, model_params: Dict = None, **data_param):
        # self.data_param = data_param
        self.m = None
        # self.y_pred = None
        # self.y_true = None
        # self.feature = self.data_param.get('feature_list', None)
        # self.train_x = self.data_param.get('feature_train', None)
        # self.train_y = self.data_param.get('label_train', None)
        # self.test_x = self.data_param.get('feature_test', None)
        # self.test_y = self.data_param.get('label_test', None)
        if evaluation is None:
            self.evaluation = Evaluation(model=self)
        else:
            self.evaluation = evaluation
        # if isinstance(params, dict):
        #     self.params.update(params)
        self.model_params = self.default.copy()
        if model_params:
            self.model_params.update(model_params)

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

    @abc.abstractmethod
    def fit(self, x, y, **options):
        """

        :param x:
        :param y:
        :param options:
        :return:
        """
        raise NotImplemented

    @abc.abstractmethod
    def predict(self, x):
        raise NotImplemented

    @staticmethod
    def convert_to_label(y_prob, thereshold=0.5):
        y_prob[y_prob >= 0.5] = 1
        y_prob[y_prob < 0.5] = 1
        return y_prob

    @abc.abstractmethod
    def save(self, save_path):
        raise NotImplemented

    @abc.abstractmethod
    def load(self, model_file):
        raise NotImplemented

    def evaluate(self, x, y, threshold=None):
        y_prob = self.predict(x)
        return self.evaluation.evaluate(y_true=y, y_pred=y_prob, threshold=threshold)

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
