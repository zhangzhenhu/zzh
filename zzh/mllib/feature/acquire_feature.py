#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
#
#
"""
模块用途描述

Authors: zhangzhenhu
Date:    2019/7/24 15:20
"""
import os
import sys
sys.path.append('../feature_code')
import numpy as np
import pandas as pd
from pyhive import hive
from hdfs import InsecureClient

from connect_hive import SparkFeature
from feature_code.history.feature import Feature


class AcquireFeature:
    def __init__(self):
        
        self.dic_path = {'otemp': '/opt/hive/warehouse/otemp.db/'}
        self.feature_result = None

    def get_url(self):
        '''
        获取高可用地址
        :return: URL
        '''
        hdfs_urls = ["http://192.168.23.223:50070", "http://192.168.24.95:50070"]
        for url in hdfs_urls:
            client = InsecureClient(url)
            try:
                client.status('/')
                return url
            except:
                continue


    def featureIsAllreadyExist(self,path):
        '''
        判断该路径下是否已经存在数据
        :param path: 路径
        :return: boolean
        '''
        url = self.get_url()
        client = InsecureClient(url, user='hdfs')
        try:
            is_exist = client.list(path)
            if is_exist:
                return True
        except:
            return False
    
    def __acq_data(self):
        """
        获取数据
        :return:
        """
        self.param1={'year':self.year,
                    'term_id':self.term_id,
                    'date':self.date,
                     'refresh':self.refresh,
                    'source':self.source}
        try:
            self.feature_code = Feature(**self.param1)

            #将pandas DataFrame 转换成spark DataFrame
            try:
                if self.sql is not None:
                    self.base_data = self.feature_code.read_sql(self.sql)
                elif self.data is not None:
                    self.base_data = self.feature_code.get_sample(self.data)
            except ValueError:
                print('传入了无效参数，data 或者 sql')

            #获取最终关联后的特征
            self.feature_result = self.feature_code.merge_feature(self.base_data,self.feature_dimension)

            #存取特征
            self.feature_code.get_result(self.feature_result,self.table_name)

            self.feature_result = self.feature_result.toPandas()
            
        finally:
            self.feature_code.stop()   
    
    def acquire_data(self,**param):
        assert param['table_name'] is not None
        
        self.param = param
        self.year = self.param.get('year',None)
        self.term_id = self.param.get('term_id',None)
        self.date = self.param.get('date','x-x-x')
        self.source = self.param.get('source','cur')
        self.data = self.param.get('data',None)
        self.sql = self.param.get('sql',None)
        self.refresh = self.param.get('refresh',False)
        self.feature_dimension = self.param.get('feature_dimension',None)
        self.label_name = self.param.get('label_name','label')
        self.table_name = self.param.get('table_name')
        self.drop_table = self.param.get('drop_table',False)
        
        file_path = self.dic_path[self.table_name.split('.')[0]] + self.table_name.split('.')[-1]
        
        if self.drop_table is False and self.refresh is False and self.featureIsAllreadyExist(file_path):
            try:
                self.connect_spark = SparkFeature()
                self.feature_result = self.connect_spark.read_data("""select * from %s"""%self.table_name)
            finally:
                self.connect_spark.stop_spark()
        else:
            self.__acq_data()
            
        return self.feature_result