"""
DeepFm的壳
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc
from time import time

from ._deep_fm import DeepFM as TfDeepFM
from .model import ABCModel

# from evaluation import Evaluation
from zzh.mllib.feature import pandas_to_numpy_fm, DataSet


class DeepFM(ABCModel):
    name = "Deepfm"
    description = "Deepfm"

    def _fit(self, dataset: DataSet, **options):
        # self.param = param
        # params = {'max_depth': range(2, 6, 1)}
        # params = {'min_child_weight': range(1, 6, 1)}
        # params = {'gamma': [0, 0.1, 0.2, 0.3]}
        # self.adjust_params(params)

        #  self.train_x = self.select_features(self.train_x)
        self.m = TfDeepFM(**self.model_params)
        self.m.init_graph()

        self.m.fit(dataset.xi, dataset.x, dataset.y)
        # print('model XGB fit begin:')
        # self.m.fit(dataset.x, dataset.y, **options)

        return self

    def fit_bak(self, **param):
        self.dfm_params = param
        print(self.dfm_params, file=sys.stderr)
        NUM_SPLITS = self.param.get('NUM_SPLITS', 3)
        RANDOM_SEED = self.param.get('RANDOM_SEED', 2019)
        self.folds = list(StratifiedKFold(n_splits=NUM_SPLITS, shuffle=True,
                                          random_state=RANDOM_SEED).split(self.Xi_train, self.y_train))

        _get = lambda x, l: [x[i] for i in l]
        self.y_test_meta = np.zeros((len(self.Xi_test), 1), dtype=float)

        for i, (train_idx, valid_idx) in enumerate(self.folds):
            Xi_train_, Xv_train_, y_train_ = _get(self.Xi_train, train_idx), _get(self.Xv_train, train_idx), \
                                             _get(self.y_train, train_idx)
            Xi_valid_, Xv_valid_, y_valid_ = _get(self.Xi_train, valid_idx), _get(self.Xv_train, valid_idx), \
                                             _get(self.y_train, valid_idx)
            dfm = DeepFM(**self.dfm_params)
            dfm.fit(Xi_train_, Xv_train_, y_train_, Xi_valid_, Xv_valid_, y_valid_)
            self.y_test_meta[:, 0] += dfm.predict(self.Xi_test, self.Xv_test)
            print('test_:', dfm.evaluate(self.Xi_test, self.Xv_test, self.y_test))

        self.test_y = self.y_test
        self.test_y_pred = self.y_test_meta / float(len(self.folds))

        return self

    def fit(self, Xi_train, Xv_train, y_train):

        # Xi_train = self.Xi_train
        # Xv_train = self.Xv_train
        # y_train = self.y_train
        self._model.init_graph()
        self._model.fit(Xi_train, Xv_train, y_train)
        # if (self.y_test is not None) & (self.Xi_test is not None) & (self.Xv_test is not None):
        #     self.test_y = self.y_test
        #     self.test_y_pred = dfm.predict(self.Xi_test, self.Xv_test)
        #     print('test_:', dfm.evaluate(self.Xi_test, self.Xv_test, self.y_test), file=sys.stderr)
        #
        return self

    def predict_bak(self, Xi, Xv, layer=3):
        """
        推断
        :param dfTest:, feat_dict:one-hot索引字典，ckpt_path：模型路径，layer：神经网络层数
        :return:
        """

        def get_batch(Xi, Xv, y, batch_size, index):
            start = index * batch_size
            end = (index + 1) * batch_size
            end = end if end < len(y) else len(y)
            return Xi[start:end], Xv[start:end], [[y_] for y_ in y[start:end]]

        # if 'target' in dfTest.columns:
        #     dfTest.drop('target', axis=1, inplace=True)

        # Xi_test, Xv_test, feature_dim = pandas_to_numpy_fm(dfTest, feature_list, one_hot)

        y_test_meta = np.zeros((Xi.shape[0], 1), dtype=float)

        with tf.Session() as sess:
            # print('load model', file=sys.stderr)
            # t1 = time()
            # saver = tf.train.import_meta_graph(ckpt_path + '.meta', clear_devices=True)
            # saver.restore(sess, ckpt_path)
            # print('----------- load model ... [%.1f s ] ' % (time() - t1), file=sys.stderr)
            feat_index = tf.get_default_graph().get_tensor_by_name('feat_index:0')
            feat_value = tf.get_default_graph().get_tensor_by_name('feat_value:0')

            # dropout_keep_fm:2 dropout_keep_deep:3 train_phase:BN,False
            dropout_keep_fm = tf.get_default_graph().get_tensor_by_name('dropout_keep_fm:0')
            dropout_keep_deep = tf.get_default_graph().get_tensor_by_name('dropout_keep_deep:0')
            train_phase = tf.get_default_graph().get_tensor_by_name('train_phase:0')

            out = tf.get_default_graph().get_tensor_by_name('out:0')

            dummy_y = [1] * len(Xi)
            batch_index = 0
            batch_size = 128
            Xi_batch, Xv_batch, y_batch = get_batch(Xi, Xv, dummy_y, batch_size, batch_index)
            y_pred = None

            while len(Xi_batch) > 0:
                num_batch = len(y_batch)
                feed_dict = {feat_index: Xi_batch,
                             feat_value: Xv_batch,
                             dropout_keep_fm: [1.0] * 2,
                             dropout_keep_deep: [1.0] * layer,
                             train_phase: False}

                # y_pred = sess.run(out, feed_dict=feed_dict)

                batch_out = sess.run(out, feed_dict=feed_dict)

                if batch_index == 0:
                    y_pred = np.reshape(batch_out, (num_batch,))
                else:
                    y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))

                batch_index += 1
                Xi_batch, Xv_batch, y_batch = get_batch(Xi, Xv, dummy_y, batch_size, batch_index)

            print(y_pred.shape, y_pred, file=sys.stderr)
            y_test_meta[:, 0] = y_pred

            return y_pred

    def predict(self, Xi, Xv):
        return self._model.predict(Xi, Xv)

    def save(self, save_path):

        self._model.save(save_path)

    def load(self, model_path):

        # model_path = model_path + '/deepfm.meta'
        # self._model = DeepFM(**self.dfm_params)
        self._model.load(model_path)

        # self.sess = tf.Session()
        # with tf.Session() as sess:
        # print('load model', file=sys.stderr)
        # t1 = time()
        # saver = tf.train.import_meta_graph(model_path + '/deepfm.meta', clear_devices=True)
        # saver.restore(self.sess, model_path)
