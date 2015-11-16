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
ss_rosl = pyrosl.ROSL( 
    method='subsample',
    sampling = (72*72,500),
    rank = 3,
    reg = 0.02,
    tol = 1E-5,
    iters = 50,
    verbose = True
)

ss_rosl.fit_transform(D)
print '---'

# Run the full ROSL algorithm
full_rosl = pyrosl.ROSL(
    method = 'full',
    rank = 3,
    reg = 0.015,
   )
print 'Starting ROSL'
full_rosl.fit_transform(D)
print 'Finished ROSL'
print '---'

# Output some numbers
error1 = np.linalg.norm(D - ss_rosl.model_ - ss_rosl.residuals_, 'fro') / np.linalg.norm(D, 'fro')
error2 = np.linalg.norm(D - full_rosl.model_ - full_rosl.residuals_, 'fro') / np.linalg.norm(D, 'fro')
error3 = np.linalg.norm(ss_rosl.model_ - full_rosl.model_, 'fro') / np.linalg.norm(full_rosl.model_, 'fro')
print 'Subsampled ROSL+ error: %.3f' % error1
print 'Full ROSL error:        %.3f' % error2
print 'ROSL/ROSL+ comparison:  %.3f' % error3
print '---'
