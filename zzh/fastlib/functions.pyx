# distutils: language = c++
cimport cython
from cpython cimport array
import array

import numpy as np
cimport numpy as np



def time_conflict(np.ndarray a, np.ndarray b,int delay):
    cdef int a_rows = a.shape[0]
    cdef int a_cols = a.shape[1]
    cdef int b_rows = b.shape[0]
    cdef int b_cols = b.shape[1]
    cdef int i=0,j=0
    for i in range(a_rows):
        for j in range(b_rows):
            if a[i][0] != b[j][0]:
                continue
            if not (a[i][1]-b[j][2] >= delay or b[j][1] - a[i][2]>=delay):
                return True
    return False




def time_conflict_2(np.ndarray a, np.ndarray b,int delay):
    cdef int a_rows = a.shape[0]
    cdef int a_cols = a.shape[1]
    cdef int b_rows = b.shape[0]
    cdef int b_cols = b.shape[1]
    cdef int i=0,j=0
    for i in range(a_rows):
        for j in range(b_rows):
            if not (a[i][0]-b[j][1] > delay or b[j][0] - a[i][1]>delay):
                return True
    return False