import os, sys, warnings
import scipy.io
import numpy as np
import pyrosl

print '---'

# Load the data
mat = scipy.io.loadmat('../Dataset.mat')
D = np.array(mat['var2'], order='F')
D = D.astype(np.double, copy = False)

# Run the sub-sampled version
print 'Starting ROSL+'
A1, E1 = pyrosl.rosl(D,'subsample', (72*72,500) , 3, 0.02, 1E-5, 50, True)
print 'Finished ROSL+'
print '---'

# Run the full ROSL algorithm
#print 'Starting ROSL'
#A2, E2 = pyrosl.rosl(D, 'full', rank=3, reg=0.015)
#print 'Finished ROSL'
#print '---'

# Output some numbers
error1 = np.linalg.norm(D - A1 - E1, 'fro') / np.linalg.norm(D, 'fro')
error2 = np.linalg.norm(D - A2 - E2, 'fro') / np.linalg.norm(D, 'fro')
error3 = np.linalg.norm(A1 - A2, 'fro') / np.linalg.norm(A2, 'fro')
print 'Subsampled ROSL+ error: %.3f' % error1
print 'Full ROSL error:        %.3f' % error2
print 'ROSL/ROSL+ comparison:  %.3f' % error3
print '---'
