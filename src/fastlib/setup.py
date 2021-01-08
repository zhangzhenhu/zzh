from setuptools import setup
# from Cython.Build import cythonize
from distutils.extension import Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import os

cur_path = os.path.dirname(__file__)

__version__ = "0.0.1"

extensions = [
    Pybind11Extension(name="fastlib",
                      sources=[os.path.join(cur_path, "bind.cpp")],
                      # Example: passing in the version to the compiled code
                      define_macros=[('VERSION_INFO', __version__)],
                      ),

]

setup(
    name='fastlib',
    install_requires=["pybind11"],
    setup_requires=["pybind11"],
    ext_modules=extensions,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
