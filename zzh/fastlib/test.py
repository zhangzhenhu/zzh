#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# 
#
"""
模块用途描述

Authors: zhangzhenhu(acmtiger@gmail.com)
Date:    2019-05-16 20:32
"""
import sys
import argparse
import numpy as np
from zzh.fastlib.functions import time_conflict, time_conflict_2

__version__ = 1.0
import time


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
    a = np.asarray([

                       [20190504, 1556933400, 1556937000] for i in range(15)
                   ] , dtype=np.int64)

    b = np.asarray([
                       [20190505, 1557019800, 1557023400] for i in range(15)
                   ], dtype=np.int64)

    aa = np.asarray([

                        [1556933400, 1556937000] for i in range(15)
                    ] , dtype=np.int64)
    bb = np.asarray([
                        [1557019800, 1557023400] for i in range(15)
                    ] , dtype=np.int64)
    t0 = time.time()

    for i in range(100000):
        time_conflict(a, b, 20)
    t1 = time.time()

    for i in range(100000):
        time_conflict_2(aa, bb, 20)
    t2 = time.time()
    print(t1 - t0, t2 - t1)


if __name__ == "__main__":

    parser = init_option()
    options = parser.parse_args()

    if options.input:

        options.input = open(options.input)
    else:
        options.input = sys.stdin
    main(options)
