import os, sys, warnings
import scipy.io
import ctypes
import numpy as np
import timeit
from numpy.ctypeslib import ndpointer

# Load the library
librosl = ctypes.cdll.LoadLibrary('./librosllib.so')
pyrosl = librosl.pyROSL

pyrosl.restype = None
pyrosl.argtypes = [ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"),
                   ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"),
                   ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"),
                   ctypes.c_int, ctypes.c_int,
                   ctypes.c_int, ctypes.c_double,
                   ctypes.c_double, ctypes.c_int,
                   ctypes.c_int, ctypes.c_int]

# Load the data
mat = scipy.io.loadmat('../../../Dataset.mat')
D = np.array(mat['var2'], order='F')
D = D.astype(np.double, copy = False)
sizeD = D.shape
m = sizeD[0]
n = sizeD[1]

A1 = np.zeros(sizeD, dtype=np.double, order='F')
E1 = np.zeros(sizeD, dtype=np.double, order='F')

pyrosl(D, A1, E1, m, n, 3, 0.015, 0.00001, 50, 0, 1000)

print np.linalg.norm(D - A1 - E1, 'fro') / np.linalg.norm(D, 'fro')


A2 = np.zeros(sizeD, dtype=np.double, order='F')
E2 = np.zeros(sizeD, dtype=np.double, order='F')

pyrosl(D, A2, E2, m, n, 3, 0.02, 0.00001, 50, 1, 1000)

print np.linalg.norm(D - A2 - E2, 'fro') / np.linalg.norm(D, 'fro')

print np.linalg.norm(A1 - A2, 'fro') / np.linalg.norm(A1, 'fro')
