import numpy as np
cimport numpy as np
import cython

cdef extern from "Gather.h":
    void cgather_batch(float*, long*, float*, int, int, int, int, int) except +
    void cgather_K(float*, long*, float*, int, int, int, int,int) except +

@cython.boundscheck(False)
def gather_batch(np.ndarray[float, ndim=3, mode="c"] raw,
           np.ndarray[long, ndim=2, mode="c"] indices,
           np.ndarray[float, ndim=2, mode="c"] result,
           int R, int B, int N, int batch_size, int n_threads):
    cgather_batch(&raw[0,0,0], &indices[0,0], &result[0,0], R, B, N, batch_size, n_threads)

@cython.boundscheck(False)
def gather_K(np.ndarray[float, ndim=3, mode="c"] raw,
           np.ndarray[long, ndim=2, mode="c"] indices,
           np.ndarray[float, ndim=2, mode="c"] result,
           int R, int B, int N, int batch_size, int n_threads):
    cgather_K(&raw[0,0,0], &indices[0,0], &result[0,0], R, B, N, batch_size, n_threads)
