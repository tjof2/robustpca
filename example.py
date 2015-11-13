import os, sys, warnings
import scipy.io
import numpy as np
import pyrosl

# Load the data
mat = scipy.io.loadmat('../Dataset.mat')
D = np.array(mat['var2'], order='F')
D = D.astype(np.double, copy = False)

A1, E1 = pyrosl.rosl(D, method='subsample', sampling=500, rank=5, reg=0.02, tol=1E-5, iters=50, verbose=False)

print np.linalg.norm(D - A1 - E1, 'fro') / np.linalg.norm(D, 'fro')

A2, E2 = pyrosl.rosl(D, method='full', sampling=-1, rank=5, reg=0.015, tol=1E-5, iters=50, verbose=False)

print np.linalg.norm(D - A2 - E2, 'fro') / np.linalg.norm(D, 'fro')


print np.linalg.norm(A1 - A2, 'fro') / np.linalg.norm(A2, 'fro')
