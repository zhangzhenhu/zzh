#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# 
#
"""
模块用途描述

Authors: zhangzhenhu(acmtiger@gmail.com)
Date:    2020/5/19 15:52
"""
import sys
import argparse
from urllib import parse
import pandas as pd

__version__ = 1.0


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


def read(fname):
    fh = open(fname)
    for line in fh:
        line = line.strip().split(' ')
        time = line[3].strip('[')
        method = line[5].strip('"')
        url = line[6]
        if method != "GET":
            continue
        if not url.startswith("/selfService/?uid="):
            continue
        result = parse.urlparse(url)
        query_dict = parse.parse_qs(result.query)
        print(query_dict, url)
        yield {'time': time,

               'uid': query_dict['uid'][0],
               'stuId': query_dict['stuId'][0],
               'from': query_dict.get('from', [None])[0],
               'url': url,
               }


def main(options):
    pass
    f0 = "/Users/zhangzhenhu/Downloads/101/united.speiyou.cn.access.log"
    f1 = "/Users/zhangzhenhu/Downloads/100/united.speiyou.cn.access.log"
    records = list(read(f0))
    records.extend(list(read(f1)))
    df = pd.DataFrame(records)
    df.to_excel("h5_nginx_log.xlsx")


if __name__ == "__main__":

    parser = init_option()
    options = parser.parse_args()

    if options.input:

        options.input = open(options.input)
    else:
        options.input = sys.stdin
    main(options)
