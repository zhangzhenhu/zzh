#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2014 Hal.com, Inc. All Rights Reserved
#
# setup.py
from setuptools import find_packages
from distutils.core import setup
# from Cython.Build import cythonize
# from distutils.extension import Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
# from Cython.Distutils import Extension
# from Cython.Distutils import build_ext
import numpy.distutils.misc_util

import os

cur_path = os.path.dirname(__file__)

__version__ = "0.0.1"

DESCRIPTION = "zzh's toolkit"
with open('README.rst') as f:
    LONG_DESCRIPTION = f.read()

include_dirs = numpy.distutils.misc_util.get_numpy_include_dirs()

NAME = "zzh"
PACKAGES = [NAME] + ["%s.%s" % (NAME, i) for i in find_packages(NAME)]
# PACKAGES = []
extensions = [
    Pybind11Extension(name="zzh.fastlib",
                      sources=[os.path.join(cur_path, "src", "fastlib", "bind.cpp")],
                      # Example: passing in the version to the compiled code
                      define_macros=[('VERSION_INFO', __version__)],
                      ),

]

# if USE_CYTHON:
#     from Cython.Build import cythonize
#
#     extensions = cythonize(extensions)

setup(
    name=NAME,
    # cmdclass={'build_ext': build_ext},
    ext_modules=extensions,
    install_requires=["pybind11"],
    setup_requires=["pybind11", ],
    include_dirs=include_dirs,
    packages=PACKAGES,
    version=__version__,  # ''版本号, 通常在 name.__init__.py 中以 __version__ = "0.0.1" 的形式被定义',
    description='工具包',
    # long_description='PyPI首页的正文',
    url='https://github.com/zhangzhenhu/zzh',
    # download_url='你项目源码的下载链接',
    # license='版权协议名',
    author='zhangzhenhu',
    author_email='acmtiger@gmail.com',
    zip_safe=False,
    cmdclass={"build_ext": build_ext},
)
