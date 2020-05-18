# distutils: language = c++
cimport cython
from cpython cimport array
import array

import numpy as np
cimport numpy as np



cdef extern from "lcs.cpp":
    pass

cdef extern from "lcs.h":
    cdef int fast_lcs[T](const T *a, int a_len, const T *b, int b_len);
    cdef cppclass LCS[T]:
        LCS() except +
        int length(T *a, int a_len, T *b, int b_lem);
        int sequence_position(int *a, int *b);


ctypedef double dtype_t

cdef void *get_pointer(np.ndarray arr):
    if arr is None:
        return NULL
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)
    # cdef double[::1] lower_view = arr
    return arr.data

cdef class pyLCS:
    cdef LCS[int] *c_object
    cdef _length
    def __cinit__(self):
        self.c_object = new LCS[int]()
        self._length = 0
    def __init__(self):
        self._length = 0
        # self.content_a = None
        # self.content_b = None

    def run(self, np.ndarray[int, ndim=1] a, np.ndarray[int, ndim=1] b):
        # print('a', a.dtype, a.dtype.itemsize, a.shape[0])
        # print("size of int",sizeof(int))
        # self.content_a = a
        # self.content_b = b
        self._length = self.c_object.length(<int*> get_pointer(a), a.shape[0], <int*> get_pointer(b), b.shape[0])
        # print('length',self._length)
        return self._length

    @property
    def length(self):
        return self._length

    def sequence_position(self):
        if self._length == 0:
            raise ValueError('No data')
            return
        a = np.zeros(self._length, dtype=np.int32)
        b = np.zeros(self._length, dtype=np.int32)

        c = self.c_object.sequence_position(<int*> get_pointer(a), <int*> get_pointer(b))
        # print('c', c)
        return a, b

    #     a.c
    #     return self.c_object.estimate(<int*> get_pointer(a), a.shape[0], <int*> get_pointer(b), b.shape[0])

    def __dealloc__(self):
        del self.c_object

def fast_length(np.ndarray a, np.ndarray b):
    # todo: 没搞清楚 np.dtype 和 c++的类型对应关系
    if a.dtype == np.int32:
        return fast_lcs[int](<int*> get_pointer(a), a.shape[0], <int*> get_pointer(b), b.shape[0])
    elif a.dtype == np.int64:
        return fast_lcs[cython.long](<cython.long*> get_pointer(a), a.shape[0], <cython.long*> get_pointer(b),
                                     b.shape[0])
    elif a.dtype == np.int8:
        return fast_lcs[char](<char*> get_pointer(a), a.shape[0], <char*> get_pointer(b), b.shape[0])
    # elif a.dtype == np.float32:
    #     return fast_lcs[float](<float*> get_pointer(a), a.shape[0], <float*> get_pointer(b), b.shape[0])
    # elif a.dtype == np.float64:
    #     return fast_lcs[double](<double*> get_pointer(a), a.shape[0], <double*> get_pointer(b), b.shape[0])
    else:
        raise TypeError("不支持的numpy类型 %s" % np.dtype)
