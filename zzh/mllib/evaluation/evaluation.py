#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
"""
模块用途描述

Authors: zhangzhenhu
Date:    2019/4/5 15:32
"""
from __future__ import print_function
from typing import List
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from IPython.display import display
from zzh.mllib.feature.dataset import DataSet


def accuracy_error(y_true, y_pred):
    return abs(y_true.mean() - y_pred.mean())


def select_valid(y_true, y_pred):
    """
    预测值可能存在空值的情况，这里剔除空值

    :param y_true:
    :param y_pred:
    :return:
    """

    # 找出非空值的位置
    valid_mask = ~np.isnan(y_pred[:, 0])

    y_pred = y_pred[valid_mask]
    y_true = y_true[valid_mask]
    return y_true, y_pred


def default_metrics():
    m = {
        "accuracy": 0,
        "tpr": 0,
        "fpr": 0,
        "precision_0": 0,
        "precision_1": 0,
        "uae": 0,
        "mae": 0,
        "mse": 0,
        "recall_0": 0,
        "recall_1": 0,
        "auc": 0,
        "f1": 0,
    }
    return m


def prob_to_binary(prob: np.ndarray, threshold):
    label = prob.copy()
    label[prob >= threshold] = 1
    label[prob < threshold] = 0
    return label


class Evaluation:
    metric_cols = [
        'name',
        'dataset',
        'coverage',  # 覆盖率,有预测结果的占比
        'uae',
        'accuracy',
        'precision_1',
        'precision_0',
        'recall_0',
        'recall_1',
        'mae',
        'mse',
        'auc',
        'f1',
        'description']

    def __init__(self, dataset: DataSet, threshold=None, name="default", ):
        self.name = name
        self.dataset = dataset
        if threshold is None:
            self.threshold = dataset.threshold
        else:
            self.threshold = threshold
        # 在某些特殊场景下，不是每条样本都能给出预测值。
        # 有些样本的预测值为空值，这里去掉空值
        self.y_true, self.y_pred = select_valid(y_true=dataset.y, y_pred=dataset.predict)
        self.y_label = prob_to_binary(self.y_pred[:, 1], self.threshold)

    @staticmethod
    def pred_binary(y_true, y_pred, threshold=0.5):

        # self.y_pred = y_pred
        y_pred_binary = y_pred.copy()
        # 找出非空值的位置
        valid_mask = ~np.isnan(y_pred_binary)

        y_pred = y_pred[valid_mask]
        y_true = y_true[valid_mask]
        return y_pred, y_true

        # y_pred_binary = y_pred_binary[valid_mask]
        y_pred_binary[y_pred_binary >= threshold] = 1
        y_pred_binary[y_pred_binary < threshold] = 0
        return y_pred_binary

        # self.y_true = y_true
        # self.y_pred_binary = y_pred_binary

    @staticmethod
    def confusion_matrix(dataset, threshold=0.5):
        pass
        # y_pred_binary = Evaluation.pred_binary(y_true, y_pred, threshold)

        # cm = metrics.confusion_matrix(y_true, y_pred_binary, labels=[1, 0])
        # return cm

    def display_confusion_matrix(self, y_true=None, y_pred=None, threshold=0.5):

        cm = self.confusion_matrix(y_true, y_pred, threshold)
        cm_pandas = pd.DataFrame(cm, columns=['pre_1', 'pre_0'],
                                 index=pd.Index(['real_1', 'real_0'], name='%s' % self.model.name))
        cm_pandas.name = self.model.name
        display(cm_pandas)

    def eval(self):

        # if dataset is None:
        #     dataset = self.dataset
        # if threshold is None:
        #     threshold = self.threshold
        # dataset = self.dataset
        # threshold = self.threshold
        mc = self.eval_binary(self.y_true, self.y_pred[:, 1], self.y_label)
        mc['coverage'] = self.y_pred.shape[0] / self.dataset.y.shape[0]
        mc['dataset'] = self.dataset.name
        mc['name'] = self.name
        return mc

    @staticmethod
    def eval_binary(y_true, y_pred, y_label):

        # raw_len = len(y_pred)
        # 在某些特殊场景下，不是每条样本都能给出预测值。
        # 有些样本的预测值为空值，这里去掉空值
        # y_true, y_pred = select_valid(y_true=y_true, y_pred=y_pred)
        # y_label = prob_to_binary(y_pred, threshold)
        mc = default_metrics()
        # mc['y_true'] = y_true
        # mc['y_prob'] = y_pred
        # mc['y_label'] = y_label

        # 全部是空值
        if len(y_pred) == 0:
            return mc

        # self.pred_binary(y_true, y_pred, threshold)

        mc['accuracy'] = metrics.accuracy_score(y_true, y_label)
        mc['tpr'] = metrics.precision_score(y_label, y_true, pos_label=1)
        try:
            mc['fpr'] = 1 - metrics.precision_score(y_label, y_true, pos_label=0)
        except:
            mc['fpr'] = 0
        mc['precision_1'] = metrics.precision_score(y_true, y_label, pos_label=1)
        mc['precision_0'] = metrics.precision_score(y_true, y_label, pos_label=0)
        mc['recall_1'] = metrics.recall_score(y_true, y_label, pos_label=1)
        mc['recall_0'] = metrics.recall_score(y_true, y_label, pos_label=0)
        mc['uae'] = accuracy_error(y_true, y_label)
        mc['mae'] = metrics.mean_absolute_error(y_true, y_pred)
        mc['mse'] = metrics.mean_squared_error(y_true, y_pred)
        try:
            mc['auc'] = metrics.roc_auc_score(y_true=y_true, y_score=y_pred)
        except ValueError:
            mc['auc'] = 0

        mc['f1'] = metrics.f1_score(y_true, y_label)

        return mc

    def print_evaluate(self, y_true=None, y_pred=None, threshold=0.5):
        from tabulate import tabulate
        ret = self.evaluate(y_true, y_pred, threshold)
        print(tabulate(pd.DataFrame([ret]), headers='keys', tablefmt='psql'))
        # for k in ['name', ] + self.key_index + ['description']:
        #     v = ret[k]
        #     if isinstance(v, float):
        #         print("%s: %0.5f" % (k, v), end=' ')
        #     else:
        #         print("%s: %s" % (k, v), end=' ')
        # print("")

    def roc_curve(self):
        return metrics.roc_curve(y_true=self.y_true, y_score=self.y_pred)

    def plot_auc(self, dataset: DataSet = None, gca=None):
        if dataset is None:
            dataset = self.dataset

        # if y_true is None and self.model is not None:
        #     y_true = self.model.y_true
        # if y_pred is None and self.model is not None:
        #     y_pred = self.model.y_pred
        # assert y_pred is not None
        # assert y_true is not None

        fpr, tpr, _ = metrics.roc_curve(y_true=dataset.y, y_score=dataset.predict)
        roc_auc = metrics.auc(fpr, tpr)
        # plt.figure()
        lw = 2
        if gca is None:
            gca = plt.figure().gca()
        gca.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        gca.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        gca.set_xlim([0.0, 1.0])
        gca.set_ylim([0.0, 1.05])
        gca.set_xlabel('False Positive Rate')
        gca.set_ylabel('True Positive Rate')
        gca.set_title(' %s roc curve' % self.name)
        gca.legend(loc="lower right")


class EvaluationPool(List):
    # key_index = ['uae', 'accuracy',
    #              'precision_1',
    #              'precision_0',
    #              'recall_0',
    #              'recall_1',
    #              'mae',
    #              'mse',
    #              'auc_score',
    #              'f1_score',
    #              'description']

    # def __init__(self, models, dataset):
    #     self.models = models
    #     self.dataset = dataset

    # def set_metrics(self,
    #                 metrics=['uae', 'accuracy', 'precision_1', 'precision_0', 'recall_0', 'recall_1', 'mae', 'mse',
    #                          'auc_score',
    #                          'f1_score', 'description']):
    #     self.metrics = metrics
    def eval(self, sort_by="auc", ascending=False, cols=None):
        records = [e.eval() for e in self]
        df = pd.DataFrame.from_records(data=records)
        df.set_index('name', inplace=True)
        df.sort_index(axis=1, )
        df.sort_values(sort_by, ascending=ascending, inplace=True)
        if cols is None:
            cols = ['dataset',
                    'coverage',
                    'auc',
                    'accuracy',
                    'precision_1',
                    'recall_1',
                    'precision_0',
                    'recall_0',
                    'mae',
                    'mse',
                    'uae',
                    'f1',
                    'tpr',
                    'fpr',
                    ]
        return df[cols]

    # def eval(self, target="test", dataset=None):
    #     # if len(self) == 0:
    #     #     return pd.DataFrame()
    #     if dataset is None:
    #         dataset = self.dataset
    #
    #     if target == 'test':
    #         x = dataset.test_x
    #         y = dataset.test_y
    #     elif target == "train":
    #         x = dataset.train_x
    #         x = dataset.train_y
    #
    #     records = [self.process_one(m, x, y) for m in self.models]
    #     df = pd.DataFrame.from_records(data=records)
    #     df.set_index('name', inplace=True)
    #     df.sort_index(axis=1, )
    #     df = df[self.key_index]
    #
    #     df.sort_values(self.key_index[0], ascending=False, inplace=True)
    #     return df
    #
    # def process_one(self, model, x, y, threshold=0.5):
    #
    #     y_prob = model.predict(x, y)
    #     ev = Evaluation(model)
    #     return ev.evaluate(y_true=y, y_pred=y_prob, threshold=threshold)
    #
    #     # if target == "train":
    #     #     records = [model.train_ev for model in self]
    #     # else:
    #     #     records = [model.test_ev for model in self]

    def plot_separated(self, target="test"):

        rows = int(ceil(len(self) / 2))
        columns = 2
        fig, axes = plt.subplots(rows, columns, figsize=(columns * 5, rows * 5))
        if target == "train":
            for i, model in enumerate(self):
                x = i // 2
                y = i % 2
                # plt.subplot(rows, columns, i)
                model.evaluation.plot_auc(y_true=model.train_y, y_pred=model.train_y_pred, gca=axes[x, y])
        else:
            for i, model in enumerate(self):
                x = i // 2
                y = i % 2
                # plt.subplot(rows, columns, i)
                model.evaluation.plot_auc(y_true=model.test_y, y_pred=model.test_y_pred, gca=axes[x, y])

    def plot_allin(self, gca=None):

        # records = [e.eval() for e in self]

        # data = []

        # for model in self.models:
        #     valid_mask = ~np.isnan(model.test_y_pred)
        #     if valid_mask.mean() == 0:
        #         data.append((None, None, None))
        #         continue
        #     score = metrics.roc_curve(y_true=model.test_y[valid_mask], y_score=model.test_y_pred[valid_mask])
        #     data.append(score)
        if not gca:
            gca = plt.figure(figsize=(10, 10)).gca()
        # roc_auc = metrics.auc(fpr, tpr)

        lw = 2
        for e in self:
            # mc = e.eval()
            fpr, tpr, _ = e.roc_curve()
            # for item, model in zip(data, self):
            #     fpr, tpr, _ = item
            if fpr is None:
                continue
            try:
                roc_auc = metrics.auc(fpr, tpr)
            except:
                roc_auc = 0
            plt.plot(fpr, tpr,  # color='darkorange',
                     lw=lw, label='%s (area = %0.2f)' % (e.name, roc_auc))

        gca.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        gca.set_xlim([0.0, 1.0])
        gca.set_ylim([0.0, 1.05])
        gca.set_xlabel('False Positive Rate')
        gca.set_ylabel('True Positive Rate')
        gca.set_title('roc curve')
        gca.legend(loc="lower right")
        plt.show()
