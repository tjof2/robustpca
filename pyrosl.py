import os, sys, warnings
import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer

# Load the library
librosl = ctypes.cdll.LoadLibrary('./libpyrosl.so.0.1')
pyrosl = librosl.pyROSL

# Python wrapper for rather obtuse C function
#
#     method   = string      - 'full'     : Use full data matrix
#                              'subsample': Use a subset of the data (ROSL+ algorithm), 
#                                            with size defined by the'sampling' option
#
#     sampling = integer     - How much of the data matrix to use for ROSL+
#
#     rank     = integer     - Initial estimate of data dimensionality
#
#     reg      = double      - Regularization parameter on l1-norm (sparse error term)
#
#     tol      = double      - Stopping criterion for iterative algorithm
#
#     iters    = integer     - Maximum number of iterations
#
#     verbose  = boolean     - Show or hide C++ output
#
def rosl(D, method='full', sampling=-1, rank=5, reg=0.01, tol=1E-5, iters=50, verbose=False):
 
    # Get size of D
    m, n = D.shape
 
    # Check for Fortran-ordered array
    if np.isfortran(D) is False:
        print 'Warning: D must be arranged in Fortran-order in memory'
        print '         Convert using numpy.asfortranarray(D)'
        return
    # Sanity-check of user parameters
    elif method == 'subsample' and sampling == -1:
        print 'Warning: Method \'subsample\' selected, but option \'sampling\' is not set'
        return
    elif method == 'subsample' and sampling > (m, n):
        print 'Warning: Method \'subsample\' selected, but option \'sampling\' is greater than D dimensions'
        return

    # Create the low-rank and error matrices    
    A = np.zeros((m, n), dtype=np.double, order='F')
    E = np.zeros((m, n), dtype=np.double, order='F')
    
    # Setup the C function
    pyrosl.restype = None
    pyrosl.argtypes = [ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"),
                       ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"),
                       ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"),
                       ctypes.c_int, ctypes.c_int,
                       ctypes.c_int, ctypes.c_double,
                       ctypes.c_double, ctypes.c_int,
                       ctypes.c_int, ctypes.c_int]
                       
    # Now run it with the users parameters
    if method == 'full':             
        pyrosl(D, A, E, m, n, rank, reg, tol, iters, 0, sampling)
    elif method == 'subsample':
        pyrosl(D, A, E, m, n, rank, reg, tol, iters, 1, sampling)
    
    # Return the results
    return (A, E)
    
    
if __name__ == "__main__":
    # There will be a test suite here
    print 'Hello, world'
    
    
    
