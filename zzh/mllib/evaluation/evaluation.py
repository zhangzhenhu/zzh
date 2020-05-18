#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
"""
模块用途描述

Authors: zhangzhenhu
Date:    2019/4/5 15:32
"""
from __future__ import print_function
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from IPython.display import display


def accuracy_error(y_true, y_pred):
    return abs(y_true.mean() - y_pred.mean())


class Evaluation:
    def __init__(self, model=None):
        self.model = model
        self.key_index = ['uae', 'accuracy', 'precision_1', 'precision_0', 'recall_0', 'recall_1', 'mae', 'mse',
                          'auc_score', 'f1_score']
        self.y_true = None
        self.y_pred = None
        self.y_pred_binary = None

    def pred_binary(self, y_true, y_pred, threshold=0.5):

        self.y_pred = y_pred
        y_pred_binary = y_pred.copy()

        valid_mask = ~np.isnan(y_pred_binary)
        y_pred = y_pred[valid_mask]
        y_pred_binary = y_pred_binary[valid_mask]
        y_true = y_true[valid_mask]
        y_pred_binary[y_pred_binary >= threshold] = 1
        y_pred_binary[y_pred_binary < threshold] = 0

        self.y_true = y_true
        self.y_pred_binary = y_pred_binary

    def confusion_matrix(self, y_true=None, y_pred=None, threshold=0.5):
        self.pred_binary(y_true, y_pred, threshold)
        cm = metrics.confusion_matrix(self.y_true, self.y_pred_binary, labels=[1, 0])
        return cm

    def display_confusion_matrix(self, y_true=None, y_pred=None, threshold=0.5):
        cm = self.confusion_matrix(y_true, y_pred, threshold)
        cm_pandas = pd.DataFrame(cm, columns=['pre_1', 'pre_0'],
                                 index=pd.Index(['real_1', 'real_0'], name='%s' % self.model.name))
        cm_pandas.name = self.model.name
        display(cm_pandas)

    #         return cm

    def evaluate(self, y_true=None, y_pred=None, threshold=0.5):
        assert y_pred is not None
        assert y_true is not None
        y_pred_binary = y_pred.copy()

        valid_mask = ~np.isnan(y_pred_binary)
        # 非空值，覆盖率
        coverage = valid_mask.mean()
        if coverage == 0:
            accuracy = 0
            tpr = 0
            fpr = 0
            precision_0 = 0
            precision_1 = 0
            uae = 0
            mae = 0
            mse = 0
            recall_0 = 0
            recall_1 = 0
            auc_score = 0
            f1_score = 0

        else:
            self.pred_binary(y_true, y_pred, threshold)

            accuracy = metrics.accuracy_score(self.y_true, self.y_pred_binary)
            tpr = metrics.precision_score(self.y_pred_binary, self.y_true, pos_label=1)
            try:
                fpr = 1 - metrics.precision_score(self.y_pred_binary, self.y_true, pos_label=0)
            except:
                fpr = 0
            precision_1 = metrics.precision_score(self.y_true, self.y_pred_binary, pos_label=1)
            precision_0 = metrics.precision_score(self.y_true, self.y_pred_binary, pos_label=0)
            recall_1 = metrics.recall_score(self.y_true, self.y_pred_binary, pos_label=1)
            recall_0 = metrics.recall_score(self.y_true, self.y_pred_binary, pos_label=0)
            uae = accuracy_error(self.y_true, self.y_pred_binary)
            mae = metrics.mean_absolute_error(self.y_true, self.y_pred)
            mse = metrics.mean_squared_error(self.y_true, self.y_pred)
            try:
                auc_score = metrics.roc_auc_score(y_true=self.y_true, y_score=self.y_pred)
            except ValueError:
                auc_score = 0

            f1_score = metrics.f1_score(self.y_true, self.y_pred_binary)
        ret = {'accuracy': accuracy,
               'coverage': coverage,
               'tpr': tpr,
               'fpr': fpr,
               'precision_1': precision_1,
               'precision_0': precision_0,
               'uae': uae,
               'mae': mae,
               'mse': mse,
               'recall_0': recall_0,
               'recall_1': recall_1,
               'auc_score': auc_score,
               'f1_score': f1_score,
               'threshold':threshold,
               }
        self.key_index = list(ret.keys())
        ret['name'] = self.model.name if self.model is not None else "NULL"
        ret['description'] = self.model.description if self.model is not None else "NULL"
        return ret

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

    def plot_auc(self, y_true=None, y_pred=None, gca=None):
        if y_true is None and self.model is not None:
            y_true = self.model.y_true
        if y_pred is None and self.model is not None:
            y_pred = self.model.y_pred
        assert y_pred is not None
        assert y_true is not None

        fpr, tpr, _ = metrics.roc_curve(y_true=y_true, y_score=y_pred)
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
        gca.set_title(' %s roc curve' % self.model.name)
        gca.legend(loc="lower right")


class EvaluationMany(list):

    def evaluate(self, target="test"):
        if len(self) == 0:
            return pd.DataFrame()

        key_index = ['uae', 'accuracy', 'precision_1', 'precision_0', 'recall_0', 'recall_1', 'mae', 'mse', 'auc_score',
                     'f1_score', 'description']
        if target == "train":
            records = [model.train_ev for model in self]
        else:
            records = [model.test_ev for model in self]

        df = pd.DataFrame.from_records(data=records)

        df.set_index('name', inplace=True)
        df.sort_index(axis=1, )
        return df[key_index].sort_values(key_index[0], ascending=False)

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

    def plot_allin(self, target="test"):

        data = []
        if target == "train":
            for model in self:
                valid_mask = ~np.isnan(model.train_y_pred)
                # 预测结果中存在空值
                if valid_mask.mean() == 0:
                    data.append((None, None, None))
                    continue
                score = metrics.roc_curve(y_true=model.train_y[valid_mask], y_score=model.train_y_pred[valid_mask])
                data.append(score)
        else:
            for model in self:
                valid_mask = ~np.isnan(model.test_y_pred)
                if valid_mask.mean() == 0:
                    data.append((None, None, None))
                    continue
                score = metrics.roc_curve(y_true=model.test_y[valid_mask], y_score=model.test_y_pred[valid_mask])
                data.append(score)

        gca = plt.figure(figsize=(10, 10)).gca()
        # roc_auc = metrics.auc(fpr, tpr)

        lw = 2
        for item, model in zip(data, self):
            fpr, tpr, _ = item
            if fpr is None:
                continue
            try:
                roc_auc = metrics.auc(fpr, tpr)
            except:
                pass
            plt.plot(fpr, tpr,  # color='darkorange',
                     lw=lw, label='%s (area = %0.2f)' % (model.name, roc_auc))

        gca.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        gca.set_xlim([0.0, 1.0])
        gca.set_ylim([0.0, 1.05])
        gca.set_xlabel('False Positive Rate')
        gca.set_ylabel('True Positive Rate')
        gca.set_title('roc curve')
        gca.legend(loc="lower right")
        plt.show()
