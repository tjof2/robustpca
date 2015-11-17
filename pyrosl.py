"""Robust Orthonormal Subspace Learning
"""

import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer

class ROSL(object):

    """Robust Orthonormal Subspace Learning Python wrapper.

    ***Full description here.***

    Parameters
    ----------
    method : string, optional
        if method == 'full' (default), use full data matrix
        if method == 'subsample', use a subset of the data with a size defined
            by the 'sampling' keyword argument (ROSL+ algorithm).

    sampling : tuple (n_cols, n_rows), required if 'method' == 'subsample'
        The size of the data matrix used in the ROSL+ algorithm.

    rank : int, optional
        Initial estimate of data dimensionality.

    reg : float, optional
        Regularization parameter on l1-norm (sparse error term).

    tol : float, optional
        Stopping criterion for iterative algorithm.

    iters : int, optional
        Maximum number of iterations.

    verbose : bool, optional
        Show or hide the output from the C++ algorithm.

    Attributes
    ----------
    model_ : array, [n_samples, n_features]
        The results of the ROSL decomposition.

    residuals_ : array, [n_components, n_features]
        The error in the model.

    """

    def __init__(self, method='full', sampling=(-1,-1), rank=5, reg=0.01, tol=1E-6, iters=500, verbose=False):

        modes = {'full':0 , 'subsample': 1}
        if method not in modes:
            raise ValueError("'method' must be one of" + modes.keys())
        self.method = method
        self._mode = modes[method]
        if method == 'subsample' and -1 in sampling:
            raise ValueError("'method' is set to 'subsample' but 'sampling' is not set.")
        self.sampling = sampling
        self.rank = rank
        self.reg = reg
        self.tol = tol
        self.iters = iters
        self.verbose = verbose
        self._pyrosl = ctypes.cdll.LoadLibrary('./librosl.so.0.2').pyROSL
        self._pyrosl.restype = ctypes.c_int
        self._pyrosl.argtypes = [
                           ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"),
                           ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"),
                           ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"),
                           ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"),
                           ctypes.c_int, ctypes.c_int,
                           ctypes.c_int, ctypes.c_double,
                           ctypes.c_double, ctypes.c_int,
                           ctypes.c_int, ctypes.c_int,
                           ctypes.c_int, ctypes.c_bool]

    def fit_transform(self, X):
        """Build a model of data X
        
        Parameters
        ----------
        X : array [n_samples, n_features]
            The data to be modelled
        
        Returns
        -------
        R : int
            The estimated rank
        
        D : array [n_samples, n_features]
            The subspace basis
            
        alpha : array [n_samples, n_features]
            The subspace coefficients
        
        E : array [n_samples, n_features]
            The error in the data model
        
        """
        X = self._check_array(X)
        n_samples, n_features = X.shape
        D = np.zeros((n_samples, n_features), dtype=np.double, order='F')
        alpha = np.zeros((n_samples, n_features), dtype=np.double, order='F') 
        E = np.zeros((n_samples, n_features), dtype=np.double, order='F')
        s1, s2 = self.sampling
        R = self._pyrosl(X, D, alpha, E, n_samples, n_features, self.rank, self.reg, self.tol, self.iters, self._mode, s1, s2, self.verbose)
        
        # This is where some trickery has to happen based on the value of R       
        self.basis_, self.coeffs_, self.residuals_ =  D, alpha, E
        return D, alpha, E

    def _check_array(self, X):
        """Sanity-checks the data and parameters.
        
        """
        x = np.copy(X)
        if np.isfortran(x) is False:
            print "Array must be in Fortran-order. Converting now."
            x = np.asfortranarray(x)
        if self.sampling > x.shape:
            raise ValueError("'sampling' is greater than the dimensions of X")
        return x



