#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# 
#
"""
模块用途描述

Authors: zhangzhenhu(acmtiger@gmail.com)
Date:    2019/1/23 14:42
"""


def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False


def is_number(uchar):
    """判断一个unicode是否是数字"""
    if uchar >= u'\u0030' and uchar <= u'\u0039':
        return True
    else:
        return False


def is_alphabet(uchar):
    """判断一个unicode是否是英文字母"""
    if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
        return True
    else:
        return False


def is_other(uchar):
    """判断是否是（汉字，数字和英文字符之外的）其他字符"""
    if not (is_chinese(uchar) or is_number(uchar) or is_alphabet(uchar)):
        return True
    else:
        return False


def char_length(uchar):
    """半角转全角"""
    inside_code = ord(uchar)
    if inside_code < 0x0020 or inside_code > 0x7e:  # 不是半角
        return 2
    return 1

def str_clength(s):
    return sum([char_length(c) for c in s])


def B2Q(uchar):
    """半角转全角"""
    inside_code = ord(uchar)
    if inside_code < 0x0020 or inside_code > 0x7e:  # 不是半角字符就返回原来的字符
        return uchar
    if inside_code == 0x0020:  # 除了空格其他的全角半角的公式为:半角=全角-0xfee0
        inside_code = 0x3000
    else:
        inside_code += 0xfee0
    return unichr(inside_code)


"""全角转半角"""


def Q2B(uchar):
    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0
    if inside_code < 0x0020 or inside_code > 0x7e:  # 转完之后不是半角字符返回原来的字符
        return uchar
    return unichr(inside_code)


"""把字符串全角转半角"""


def stringQ2B(ustring):
    return "".join([Q2B(uchar) for uchar in ustring])


"""将UTF-8编码转换为Unicode编码"""


def convert_toUnicode(string):
    ustring = string
    if not isinstance(string, unicode):
        ustring = string.decode('UTF-8')
    return ustring
