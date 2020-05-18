# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-06-24 23:03
# @Author  : zhangzhenhu
# @File    : extractor.py
import sys
import argparse
from feature.core.his_feature import HisDB
from feature.core.current_feature import CurDB
from feature.core.spark_con import AbstrctSpark, AbstrctHive, get_spark, get_hive,close_spark
import os
import pandas as pd

__version__ = 1.0
# os.environ['PYTHON_HOME'] = "./python/bin/python3"
# os.environ['LD_LIBRARY_PATH'] += ':./python/lib/'

table_dict = {
    "cur_student_class_detail": "当期学生-班级维度明细",
    "cur_teacher_term_summary": "当期教师-学期维度统计信息",
    "cur_class_detail": "当期班级统计信息",
    "cur_student_term_subject": "计算 学生 当前学期 学科 维度的特征",
    "cur_student_summary_history": "学生维度汇总（当期+历史，部分学科）",
    "cur_student_term_cumulation": "学生 学期 维度 的累计特征，不包括当前学期,",
    "cur_student_term_summary_history": "学生学期维度汇总（当期+历史，学期内各个学科就读情况）",
    "cur_teacher_summary_history": "教师维度汇总（仅历史，教师近n学期情况）",
    "cur_student_book_term_subject": "按照学期-学科维度统计学生预报名下学期班级数量",

    "his_student_class_detail": "历史学生-班级维度明细",
    "his_teacher_term_summary": "历史教师-学期维度统计信息",
    "his_student_term_summary": "历史学生学期维度汇总信息（主要包含一学期内各个学科就读情况）",
    "his_class_detail": "历史班级统计信息",
    "his_student_term_subjects_summary_long": "历史学生-学期-学科维度统计信息（长期班）",
    "his_student_term_subjects_summary_short": "历史学生-学期-学科维度统计信息（短期班）",
}


class FeatureExtractor:
    """
    提取当期特征，根据维度参数决定提取学员维度特征、班级维度特征或者是老师维度特征

    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.year = self.kwargs.get('year', 2019)
        self.term_id = self.kwargs.get('term_id', None)
        self.date = self.kwargs.get('date', 'x-x-x')
        self.date_str = '_' + ''.join(self.date.split('-'))
        self.source = self.kwargs.get('source', 'cur')
        self.refresh = self.kwargs.get('refresh', False)
        # if self.source == 'his':
        self._his_db = HisDB(**self.kwargs)
        # elif self.source == 'cur':
        self._cur_db = CurDB(**self.kwargs)
        self._result = None

    def table_list(self):

        # for k, v in sorted(table_dict.items(), key=lambda x: x[0]):
        #     print(k, v)
        return pd.DataFrame(sorted(table_dict.items(), key=lambda x: x[0]), columns=['表名', "说明"])

    #         self.__check_param()
    def _add_join(self, left_data, right_table, col_names, left_on=[], right_on=[], how="left", where=""):

        # right_data = get_spark().read_parquet(right_table)
        right_data = get_spark().table(right_table)
        # 先进行where 过滤行
        if where:
            right_data = right_data.where(where)

        # 支持重命名列名
        # new_col_names = []
        # rename_list = []
        select_cols = []

        for col in col_names:
            col = col.lower().strip()
            if ' as ' in col.lower():
                col, n_name = col.split(' as ', 1)
                col = col.strip()
                n_name = n_name.strip()
                # rename_list.append((col, n_name))
                select_cols.append(right_data[col].alias(n_name))
            else:
                select_cols.append(right_data[col])
            # new_col_names.append(col)

        # col_names = new_col_names

        # if "*" in col_names:
        #     pass
        # else:
        #     right_data = right_data[col_names + right_on]

        cond = [left_data[l] == right_data[r] for l, r in zip(left_on, right_on)]
        result = left_data.join(right_data, on=cond, how=how).select(left_data['*'], *select_cols)
        # for col in right_on:
        #     left_data = left_data.drop(right_data[col])
        # for o_name, n_name in rename_list:
        #     left_data = left_data.withColumnRenamed(o_name, n_name)
        # left_data.drop(*right_on)
        return result

    def set_label(self, data: pd.DataFrame):
        import pyspark.sql as ps
        if isinstance(data, pd.DataFrame):
            self._result = get_spark().pandas_to_spark(data)
        elif isinstance(data, ps.DataFrame):
            self._result = data
        else:
            raise ValueError("data is not valid")

    def get_feature(self, table_name, cols=[], left_on=[], right_on=[],
                    how="left", where=""):

        global table_dict
        self._result = self._add_join(self._result, table_name, cols, left_on, right_on, how, where)

    def run_cur(self, year, term_id, date):

        self._cur_db.extractor(year, term_id, date)

    def get_result_pandas(self):
        return self._result.toPandas()

    def save_result_to_table(self, table, format=None, mode=None, **options):

        self._result.write.saveAsTable(table, format=format, mode=mode, **options)

    @property
    def result(self):
        return self._result

    # def save_result_to_hdfs(self, path):
    #
    #     self._result.write.parquet

    # self._result.write.parquet(path)

    def __check_param(self):
        """
        用于参数校验
        :return:
        :return:
        """
        pass

    def read_sql(self, sql_str):

        #         self.db.read_data(sql_str)
        return get_spark().connect(sql_str)

    def read_parquet(self, sql_str):

        #         self.db.read_data(sql_str)
        return get_spark().read_parquet(sql_str)

    def get_sample(self, data):

        spark_df = self.db.engine.pandas_to_spark(data)
        return spark_df

    # def get_feature(self):
    #     """
    #     对输入的特定维度特征进行提取,
    #     :return:
    #     """
    #     self.db.extractor()

    def stop(self):
        self.db.close_engine()

    def hive_read(self, sql_str):

        hive_conn = AbstrctHive()
        return hive_conn.connect(sql_str)

    def get_result(self, data, table_name):

        self.db.engine.save_as_table(table_name, data)

    def merge_feature(self, data, table_col):
        data.show(2)

        table_dict = {
            "cur_student_term": "otemp.student_term_sum",
            "cur_student_term_subject": "otemp.student_term_subject_sum",
            "cur_teacher_term": "otemp.teacher_term_sum",
            "cur_class": "otemp.class_his",
            "cur_student": "otemp.student_his",
            "cur_teacher": "otemp.teacher_his",
            "his_teacher": "otemp.teacher_his",
            "his_student": "otemp.student_his",
            "his_class": "otemp.class_his",
            "his_student_term": "otemp.student_his_term",
        }

        if self.refresh == True:
            self.get_feature()

        for i in table_col:
            data = self.add_join(data, table_dict[i['table']], i['col'], i['on'], i['how'])
        return data

    def his_term_feature(self, year, term, col_names):

        student_term = self.db.engine.read_parquet('otemp.student_his_term')
        #         student_term.show(2)
        if col_names == "*":
            data = student_term.where("cla_year = %d and cla_term_id = %d" % (year, term))
        else:
            data = student_term[[col_names]].where("cla_year = %s and cla_term_id = %s" % (year, term))
        data.show(2)
        return data

    def his_detail(self, year, term, col_names):

        #         print(term,year)
        student_term = self.db.engine.read_parquet('otemp.student_his')
        #         student_term.show(2)
        if col_names == "*":
            data = student_term.where("cla_year = %d and cla_term_id = %d" % (year, term))
        else:
            data = student_term[[col_names]].where("cla_year = %d and cla_term_id = %d" % (year, term))
        data.show(2)
        return data

    def current_student_term_Cum(self, col_names):
        cnt_path = 'otemp.student_his_cnt_term_' + str(self.year) + '_0' + str(self.term_id) + '_'
        data = self.db.engine.read_parquet(cnt_path)
        if col_names != "*":
            data = data[[col_names]]

        data.show(2)
        return data

    def current_student_detail_Cum(self, col_names):
        cnt_path = 'otemp.student_his_cnt_new_' + str(self.year) + '_0' + str(self.term_id) + '_'
        data = self.db.engine.read_parquet(cnt_path)
        if col_names != "*":
            data = data[[col_names]]

        data.show(2)
        return data


def init_option():
    """
    初始化命令行参数项
    Returns:
        OptionParser 的parser对象
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input",
                        help=u"输入文件；默认标准输入设备")
    parser.add_argument("-o", "--output", dest="output",
                        help=u"输出文件；默认标准输出设备")
    return parser


def main(options):
    pass


if __name__ == "__main__":

    parser = init_option()
    options = parser.parse_args()

    if options.input:

        options.input = open(options.input)
    else:
        options.input = sys.stdin
    main(options)
