#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-07-09 20:41
# @Author  : zhangzhenhu
# @File    : connect_hive.py

import pandas as pd
import numpy as np
from datetime import datetime
from pyhive import hive

import os
import sys
import time, datetime

# os.environ['SPARK_HOME']='/usr/local/spark-2.1.1-bin-hadoop2.6'
# os.environ['JAVA_HOME']='/usr/local/jdk1.8.0_101'

import findspark

findspark.init()

from pyspark.sql import SparkSession

dic_path = {
    'otemp': '/opt/hive/warehouse/otemp.db/'
}


class SparkFeature:

    def __init__(self):
        builder = SparkSession.builder
        builder.config("spark.executor.memory", '2G')
        builder.config("spark.driver.memory", '6G')
        builder.config("spark.executor.instances", 31)
        builder.config("spark.executor.cores", 3)
        builder.config("spark.default.parallelism", 600)
        builder.config("spark.sql.shuffle.partitions", 600)
        # builder.config("spark.storage.memoryFraction",0.9)
        # builder.config("spark.shuffle.memoryFraction",0.8)
        self.spark = builder.enableHiveSupport().appName("read_feature").getOrCreate()

        self.stu_summer_regist_data = None
        self.base_data = None

        # self.stu_summer_feature_file = 'data_mining/refund_before_class/data/feature_data/stu_summer_feature.pkl'

    def connect_spark(self, sql):
        data = self.spark.sql(sql)
        print(data.count())
        return data

    def read_parquet(self, db_file):
        """
        读取parquet
        :return:
        """
        file_path = dic_path[db_file.split('.')[0]] + db_file.split('.')[-1]
        data = self.spark.read.parquet(file_path)
        return data

    def read_data(self, sql_str):
        lst = sql_str.split(' ')
        tb_name = lst[lst.index('from') + 1]
        self.read_parquet(tb_name).registerTempTable(tb_name.split('.')[-1])
        lst[lst.index('from') + 1] = tb_name.split('.')[-1]
        sql = ' '.join(lst)
        data = self.connect_spark(sql)
        return data.toPandas()

    def stop_spark(self):
        self.spark.stop()


class HiveData:
    def __init__(self):
        self.data = None
        self.sql = None
        self.data_path = None

    def __init_connection(self, host="192.168.23.223", port=10000):
        hive_conn = hive.Connection(host=host, port=port)
        hive_cursor = hive_conn.cursor()
        return hive_cursor

    def __query_data(self, hive_cursor, sql):
        hive_cursor.execute(sql)
        result = hive_cursor.fetchall()
        columns = [col[0].split('.')[-1] for col in hive_cursor.description]
        self.data = pd.DataFrame([list(row) for row in result], columns=columns)

    def connect(self, data_path, sql):
        self.data_path = data_path
        self.sql = sql

        hive_cursor = self.__init_connection()
        self.__query_data(hive_cursor, self.sql)
        self.__save_data()
        return self.data

    def connect_table(self, sql):
        self.sql = sql
        hive_cursor = self.__init_connection()
        self.__query_data(hive_cursor, self.sql)
        return self.data

    def __save_data(self):
        self.data.to_pickle(self.data_path)

#     def __hive_sql(self):
#         self.sql = """SELECT * FROM otemp.stu_history_feature"""
