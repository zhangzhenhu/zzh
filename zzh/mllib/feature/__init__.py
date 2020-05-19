#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019-07-09 19:21
# @Author  : zhangzhenhu
# @File    : feature.py


import os
import sys
import random
import numpy as np
import pandas as pd
from scipy.special import expit as sigmod
from pandas.api.types import is_object_dtype
from sklearn.model_selection import train_test_split

from .dataset import DataSet


def pandas_to_numpy_ml(df_data: pd.DataFrame, feature_list: list, one_hot: dict):
    """
    col1 = (v1,v2,v3)

    pandas_to_numpy_ml(df_data,["col1","col2",... ],{'col1':(v1,v2,v3), 'col2':(v1,v2,v3,v4)})
    :param df_data:
    :param feature_list: df_data的col_name的列表,["col1","col2",... ]
    :param one_hot: 需要进行one hot 的列 {'col1':(v1,v2,v3), 'col2':(v1,v2,v3,v4)}
    :return:
    """
    col_n = 0
    for fe in feature_list:
        if fe in one_hot:
            col_n += len(one_hot[fe])
        else:
            col_n += 1
    shape = (df_data.shape[0], col_n)

    header = list()
    numpy_data = np.zeros(shape, dtype=np.float32)

    j = 0
    for fe in feature_list:
        if fe in one_hot:
            # pass
            # 填充
            dd = {v: i for i, v in enumerate(one_hot[fe])}

            for _, v in enumerate(one_hot[fe]):
                header.append(f"{fe}_{v}")

            for i, v in enumerate(df_data[fe]):

                if v not in one_hot[fe]:
                    continue
                else:
                    numpy_data[i, j + dd[v]] = 1
            j += len(one_hot[fe])
        else:
            numpy_data[:, j] = df_data.loc[:, fe]
            j += 1
            header.append(f"{fe}")

    return numpy_data, header


def pandas_to_numpy_fm(df_data: pd.DataFrame, feature_list: list, one_hot: dict):
    """
    pandas DataFrame 数据转换成模型的numpy特征矩阵，

    :param df_data:
    :param feature_list:
    :param one_hot:
    :return:
    """
    # col_n = 0
    # for fe in feature_list:
    #     if fe in one_hot:
    #         col_n += len(one_hot[fe])
    #     else:
    #         col_n += 1
    shape = (df_data.shape[0], len(feature_list))

    xi = np.zeros(shape)
    xv = np.zeros(shape)

    feature_index = 0
    for field_index, fe in enumerate(feature_list):
        if fe in one_hot:
            xv[:, field_index] = 1

            value_map = {v: i for i, v in enumerate(one_hot[fe])}

            # 对one-hot中某个特征新出现的值，xi取当前列索引位，对应的xv取0
            # 理论上对应的做embedding之后全部为0
            xi[:, field_index] = df_data[fe].map(lambda v: value_map.get(v, np.nan)) + feature_index

            invalid = np.isnan(xi[:, field_index])
            # 如果 xi 中存在空值，则将空值对应位置的的 xv 的值改为0
            if np.any(invalid):
                xv[invalid, field_index] = 0
                xi[invalid, field_index] = feature_index

            # for i, v in enumerate(df_data[fe]):
            #     if v not in one_hot[fe]:
            #         xi[field_index] = feature_index + 0
            #         xv[i, field_index] = 0
            #     else:
            #         xi[field_index] = feature_index + value_map[v]

            feature_index += len(one_hot[fe])
        else:

            xi[:, field_index] = feature_index
            xv[:, field_index] = df_data.loc[:, fe]
            feature_index += 1
    return xi, xv, feature_index


def gen_feat_dict(df_train, df_test, numeric_cols=[], ignore_cols=[]):
    dfTrain = df_train
    dfTest = df_test
    if dfTrain is None and dfTest is not None:
        print('dfTest is not None dfTrain is None', file=sys.stderr)
        df = dfTest
    elif dfTest is None and dfTrain is not None:
        print('dfTrain is not None dfTest is None', file=sys.stderr)
        df = dfTrain
    elif dfTrain is None and dfTest is None:
        print('ERROR: Data is None!', file=sys.stderr)
    else:
        print('dfTrain and dfTest concat', file=sys.stderr)
        df = pd.concat([dfTrain, dfTest], sort=False)

    feat_dict = {}
    tc = 0

    for col in df.columns:
        if col in ignore_cols:
            continue
        if col in numeric_cols:
            feat_dict[col] = tc
            tc = tc + 1
        else:
            us = df[col].unique()
            feat_dict[col] = dict(zip(us, range(tc, len(us) + tc)))
            tc = tc + len(us)
    feat_dim = tc

    return feat_dict, feat_dim


def pandas_to_numpy_fm_v2(df_data: pd.DataFrame, numeric_cols: list,
                          ignore_cols: list, feat_dict: dict, has_label=False):
    dfi = df_data.copy()
    if has_label:
        y = dfi['target'].values.tolist()
        dfi.drop('target', axis=1, inplace=True)
    dfv = dfi.copy()

    for col in dfi.columns:
        if col in ignore_cols:
            dfi.drop(col, axis=1, inplace=True)
            dfv.drop(col, axis=1, inplace=True)
            continue
        if col in numeric_cols:
            dfi[col] = feat_dict[col]
            # 连续值特征归一化
            d = dfv[col].max() - dfv[col].min()
            if d == 0:
                dfv[col] = 0
            else:
                dfv[col] = (dfv[col] - dfv[col].min()) / d
        else:
            dfi[col] = dfi[col].map(feat_dict[col])
            dfv[col] = 1

    # list of list of feature indices of each sample in the dataset
    Xi = dfi.values.tolist()
    # list of list of feature values of each sample in the dataset
    Xv = dfv.values.tolist()

    if has_label:
        return Xi, Xv, y,
    else:
        return Xi, Xv,


def sample_info(df_data: pd.DataFrame, group_by):
    df_info = df_data.groupby(by=group_by).agg({'stu_id': 'count'})
    df_info.rename(columns={'stu_id': "数量"}, inplace=True)
    s = df_info['数量'].sum()
    df_info["占比"] = df_info['数量'] / s
    df_info["总计"] = s
    #     pd.concat([df_info)

    return df_info


def normalization(df_data: pd.DataFrame, cols):
    """
    对列进行最大值最小值归一化
    :param df_data:
    :param cols:
    :return:
    """
    for col in cols:
        _min = df_data[col].min()
        _max = df_data[col].max()
        interval = _max - _max
        df_data[col].apply(lambda x: (x - _min) / interval)

    return df_data


def na_info(df_data, cols: list = []):
    '''
    查看特征数据分布以及空值比
    :param df_data: 选择需要查看的数据集分布
    :param cols: 指定要查看的列
    :return:dataframe
    '''
    data_info = df_data.describe().transpose()
    if not cols:
        feature_name = [column for column in df_data.columns]
    else:
        feature_name = cols
    fea_ = []
    pro_ = []
    null_ = []
    for fea in feature_name:
        fea_.append(fea)
        null_num = df_data[fea].isnull().sum()
        null_.append(null_num)
        pro = round((null_num * 1.0) / df_data.shape[0], 2)
        pro_.append(pro)
    dict_na = {'feature': fea_,
               '空值比例': pro_,
               '空值数量': null_
               }
    dict_na = pd.DataFrame(dict_na).set_index('feature')
    data_info = data_info.join(dict_na, how='inner')
    return data_info


class FeatureData:

    def __init__(self, df_train: pd.DataFrame = None, df_test: pd.DataFrame = None):
        self.data = None
        self.label_name = 'label'
        self.data_feature = None  # 废弃
        self.data_label = None  # 废弃

        self.feature_train = None  # 废弃
        self.feature_test = None  # 废弃
        self.label_train = None  # 废弃
        self.label_test = None  # 废弃
        self.data_path = None  # 废弃
        self.feature_result = None

        self.df_train = df_train  # 训练集的 pandas 数据
        self.df_test = df_test  # 测试集的 pandas 数据

        # 这两个值,自己完成特征工程后，手动存储在这两个变量上即可
        self.mt_train = None  # 训练集的 numpy 数据,可以输入给模型的
        self.mt_test = None  # 测试集的 numpy 数据,可以输入给模型的

        # concat(df_train,df_test,axis=0) 之后的结果，但不会自动生成，
        # 有需要时自己通过
        self.df_all = None
        # self.sql = None
        # self.connect_hive = HiveData()

    def save(self, save_path='./feature_cache'):
        os.makedirs(save_path, exist_ok=True)

        self.df_train.to_pickle(os.path.join(save_path, 'df_train.pkl'))
        self.df_test.to_pickle(os.path.join(save_path, 'df_test.pkl'))
        if self.mt_test:
            np.save(os.path.join(save_path, 'mt_train.pkl'), self.mt_train)

        if self.mt_test:
            np.save(os.path.join(save_path, 'mt_test.pkl'), self.mt_test)

    def load(self, load_path='./feature_cache'):

        if os.path.exists(os.path.join(load_path, 'df_train.pkl')):
            self.df_train = pd.read_pickle(os.path.join(load_path, 'df_train.pkl'))

        if os.path.exists(os.path.join(load_path, 'df_test.pkl')):
            self.df_test = pd.read_pickle(os.path.join(load_path, 'df_test.pkl'))

        if os.path.exists(os.path.join(load_path, 'mt_train.pkl')):
            self.mt_train = np.load(os.path.join(load_path, 'mt_train.pkl'))

        if os.path.exists(os.path.join(load_path, 'mt_test.pkl')):
            self.mt_test = np.load(os.path.join(load_path, 'mt_test.pkl'))

    # def __check_param(self):
    #     pass
    #
    # def __get_data(self):
    #     """
    #     从hive获取数据
    #     :return:
    #     """
    #
    #     if os.path.exists(self.data_path):
    #         self.data = pd.read_pickle(self.data_path)
    #     else:
    #         self.data = self.connect_hive.connect(self.data_path, self.sql)

    # def obtain_data(self, **param):
    #     self.param = param
    #     self.label_name = self.param.get('label_name', 'label')
    #     self.sql = self.param.get('sql', None)
    #     self.data_path = self.param.get('data_path')
    #     self.__get_data()
    #
    #     self.data_label = self.data[self.label_name].copy()
    #     self.data_feature = self.data.drop(self.label_name, axis=1).copy()

    # def split_feat_label(self):
    #     self.data_label = self.feature_result[self.label_name].copy()
    #     self.data_feature = self.feature_result.drop(self.label_name, axis=1).copy()
    #
    # def select_features(self, feature_list, drop=True, descrbe=None):
    #     if descrbe == 'train':
    #         if drop:
    #             self.feature_train = self.feature_train.drop(feature_list, axis=1)
    #         else:
    #             self.feature_train = self.feature_train[feature_list]
    #     elif descrbe == 'test':
    #         if drop:
    #             self.feature_test = self.feature_test.drop(feature_list, axis=1)
    #         else:
    #             self.feature_test = self.feature_test[feature_list]
    #     else:
    #         if drop:
    #             self.data_feature = self.data_feature.drop(feature_list, axis=1)
    #         else:
    #             self.data_feature = self.data_feature[feature_list]

    # def one_hot_encoder_1(self, nan_as_category=True):
    #     original_columns = list(self.data_feature.columns)
    #     categorical_columns = [col for col in self.data_feature.columns if self.data_feature[col].dtype == 'object']
    #     self.data_feature = pd.get_dummies(self.data_feature, columns=categorical_columns, dummy_na=nan_as_category)

    def one_hot_ml(self, data, one_hot_columns, nan_as_category=True):
        """
        传统的one hot编码
        :param data:
        :param one_hot_columns:
        :param nan_as_category:
        :return:
        """
        if data == 'train':
            self.feature_train = pd.get_dummies(self.feature_train, columns=one_hot_columns, dummy_na=nan_as_category)
        elif data == 'test':
            self.feature_test = pd.get_dummies(self.feature_test, columns=one_hot_columns, dummy_na=nan_as_category)
        else:
            self.data_feature = pd.get_dummies(self.data_feature, columns=one_hot_columns, dummy_na=nan_as_category)

    def gen_feat_dict(self):
        """
        计算特征--索引位映射字典
        :param data:
        :param one_hot_columns:
        :param nan_as_category:
        :return:
        """
        # self.numeric_cols = numeric_cols
        # self.ignore_cols = ignore_cols
        dfTrain = self.df_train

        dfTest = self.df_test
        if dfTrain is None and dfTest is not None:
            print('dfTest is not None dfTrain is None', file=sys.stderr)
            df = dfTest
        elif dfTest is None and dfTrain is not None:
            print('dfTrain is not None dfTest is None', file=sys.stderr)
            df = dfTrain
        elif dfTrain is None and dfTest is None:
            print('ERROR: Data is None!', file=sys.stderr)
        else:
            print('dfTrain and dfTest concat', file=sys.stderr)
            df = pd.concat([dfTrain, dfTest], sort=False)

        self.feat_dict = {}
        tc = 0

        for col in df.columns:
            if col in self.ignore_cols:
                continue
            if col in self.numeric_cols:
                self.feat_dict[col] = tc
                tc = tc + 1
            else:
                us = df[col].unique()
                self.feat_dict[col] = dict(zip(us, range(tc, len(us) + tc)))
                tc = tc + len(us)
        self.feat_dim = tc  # one-hot之后的维度

    def one_hot_parse(self, df=None, has_label=False, numeric_cols=[], ignore_cols=[]):
        """
        one-hot稀疏处理
        :param df:
        :param has_label:
        :return:
        """
        self.numeric_cols = numeric_cols
        self.ignore_cols = ignore_cols

        self.gen_feat_dict()
        dfi = df.copy()
        if has_label:
            y = dfi['label'].values.tolist()
            dfi.drop('label', axis=1, inplace=True)
        dfv = dfi.copy()

        for col in dfi.columns:
            if col in self.ignore_cols:
                dfi.drop(col, axis=1, inplace=True)
                dfv.drop(col, axis=1, inplace=True)
                continue
            if col in self.numeric_cols:
                dfi[col] = self.feat_dict[col]
                # 连续值特征归一化
                # d = dfv[col].max() - dfv[col].min()
                # if d == 0:
                #     dfv[col] = 0
                # else:
                #     dfv[col] = (dfv[col] - dfv[col].min()) / d
            else:
                dfi[col] = dfi[col].map(self.feat_dict[col])
                dfv[col] = 1

        # list of list of feature indices of each sample in the dataset
        Xi = dfi.values.tolist()
        # list of list of feature values of each sample in the dataset
        Xv = dfv.values.tolist()

        if has_label:
            return Xi, Xv, y,
        else:
            return Xi, Xv,

    def retype_features(self, data_type='int'):
        for column in self.data_feature.columns:
            if is_object_dtype(self.data_feature[column]):
                self.data_feature[column] = self.data_feature[column].astype(data_type)

        return self.data

    # def na_pro(self, descrbe=None):
    #     '''
    #     查看特征数据分布以及空值比
    #     :param descrbe: 选择需要查看的数据集分布
    #     :return:dataframe
    #     '''
    #     if descrbe == 'train':
    #         data = self.feature_train
    #     elif descrbe == 'test':
    #         data = self.feature_test
    #     else:
    #         data = self.data_feature
    #     data_info = data.describe().transpose()
    #     feature_name = [column for column in data.columns]
    #     fea_ = []
    #     pro_ = []
    #     null_ = []
    #     for fea in feature_name:
    #         fea_.append(fea)
    #         null_num = data[fea].isnull().sum()
    #         null_.append(null_num)
    #         pro = round((null_num * 1.0) / data.shape[0], 2)
    #         pro_.append(pro)
    #     dict_na = {'feature': fea_,
    #                '空值比例': pro_,
    #                '空值数量': null_
    #                }
    #     dict_na = pd.DataFrame(dict_na).set_index('feature')
    #     data_info = data_info.join(dict_na, how='inner')
    #     return data_info
    #
    # def fill_na(self, descrbe=None):
    #     """
    #     填充空值。  貌似可以不需要，直接自己操作 df_*
    #     :param descrbe:
    #     :return:
    #     """
    #     if descrbe == 'train':
    #         data = self.feature_train
    #     elif descrbe == 'test':
    #         data = self.feature_test
    #     else:
    #         data = self.data_feature
    #     data.fillna(0, inplace=True)

    def split_data(self, test_size=0.3, random_state=42):
        """
        需要重写
        :param test_size:
        :param random_state:
        :return:
        """
        self.feature_train, self.feature_test, self.label_train, self.label_test = \
            train_test_split(self.data_feature, self.data_label, test_size=test_size, random_state=random_state)

    # def __data_statistics(self, ds_label):
    #     _vc = ds_label.value_counts().to_dict()
    #     _total = sum(_vc.values())
    #
    #     return {"数量总量": _total,
    #             '正样本数': _vc[1],
    #             '负样本数': _vc[0],
    #             '正样本比例': _vc[1] / float(_total),
    #             }

    # def describe_s(self, is_split=True):
    #     '''
    #     查看数据集的正负样本比例
    #     :param is_split: 控制展示的数据集，is_split=True展示划分数据集后的样本比例，False展示全量数据集的正负样本比例
    #     :return:dataframe
    #     '''
    #     if is_split:
    #         record1 = self.__data_statistics(self.label_test)
    #         record1['name'] = "测试数据"
    #         record2 = self.__data_statistics(self.label_train)
    #         record2['name'] = "训练数据"
    #         df_ret = pd.DataFrame([record1, record2])
    #         df_ret.set_index('name', inplace=True)
    #     else:
    #         record = self.__data_statistics(self.data_label)
    #         record['name'] = '全部数据集'
    #         df_ret = pd.DataFrame([record])
    #         df_ret.set_index('name', inplace=True)
    #     return df_ret

    def drop_columns(self, cols):
        """
        删除一些列
        :param cols:
        :return:
        """

        self.df_train.drop(cols, axis=1, inplace=True)
        self.df_test.drop(cols, axis=1, inplace=True)

    def make_df_all(self):
        """
        注意每个都是重新生成新的
        :return:
        """
        self.df_all = pd.concat([self.df_train, self.df_test], axis=0)

    def split_df_all(self, by='_type', df_all: pd.DataFrame = None):
        """
        把 df_all 重新拆解成df_train 和df_test。
        :param by: 指定按照那一列拆分
        :param df_all: 可以传入需要拆分的数据
        :return:
        """
        if df_all is None:
            df_all = self.df_all
        self.df_train = df_all[df_all[by] == 'train'].copy()
        self.df_test = df_all[df_all[by] == 'test'].copy()

    def na_info(self, data="train", cols: list = []):
        """
        查看空值信息
        :param data:
        :param cols:
        :return:
        """
        if data == "train":
            na_info(self.df_train, cols)
        elif data == "test":
            na_info(self.df_test, cols)
        elif data == "all":
            na_info(self.df_all(), cols)
        else:
            raise ValueError('Invalid data %s' % data)
