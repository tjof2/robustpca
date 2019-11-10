# -*- coding: utf-8 -*-
# Copyright 2015-2019 Tom Furnival
#
# This file is part of RobustPCA.
#
# RobustPCA is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RobustPCA is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with RobustPCA.  If not, see <http://www.gnu.org/licenses/>.

import ctypes, os
import numpy as np
from numpy.ctypeslib import ndpointer


class ROSL(object):

    """ Robust Orthonormal Subspace Learning Python wrapper.

    Robust Orthonormal Subspace Learning (ROSL) seeks to recover a low-rank matrix A
    and a sparse error matrix E from a corrupted observation X:

        min ||A||_* + lambda ||E||_1    subject to X = A + E

    where ||.||_* is the nuclear norm, and ||.||_1 is the l1-norm. ROSL further models
    the low-rank matrix A as spanning an orthonormal subspace D with coefficients alpha

        A = D*alpha

    Further information can be found in the paper:

        X Shu, F Porikli, N Ahuja. (2014) "Robust Orthonormal Subspace Learning:
        Efficient Recovery of Corrupted Low-rank Matrices"
        http://dx.doi.org/10.1109/CVPR.2014.495

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

    def __init__(
        self,
        method="full",
        sampling=(-1, -1),
        rank=5,
        reg=0.01,
        tol=1e-6,
        iters=500,
        verbose=True,
    ):

        modes = {"full": 0, "subsample": 1}
        if method not in modes:
            raise ValueError("'method' must be one of" + modes.keys())
        self.method = method
        self._mode = modes[method]
        if method == "subsample" and -1 in sampling:
            raise ValueError(
                "'method' is set to 'subsample' but 'sampling' is not set."
            )
        self.sampling = sampling
        self.rank = rank
        self.reg = reg
        self.tol = tol
        self.iters = iters
        self.verbose = verbose
        libpath = os.path.dirname(os.path.abspath(__file__)) + "/librosl.so.0.2"
        self._pyrosl = ctypes.cdll.LoadLibrary(libpath).pyROSL
        self._pyrosl.restype = ctypes.c_int
        self._pyrosl.argtypes = [
            ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"),
            ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"),
            ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"),
            ndpointer(ctypes.c_double, flags="F_CONTIGUOUS"),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_bool,
        ]
        self.components_ = None

    def fit(self, X):
        """ Build a model of data X

        Parameters
        ----------
        X : array [n_samples, n_features]
            The data to be modelled.

        Returns
        -------
        self : object
            Returns the instance itself.

        """

        self._fit(X)

        return self

    def fit_transform(self, X):
        """ Build a model of data X and apply it to data X

        Parameters
        ----------
        X : array [n_samples, n_features]
            The data to be modelled.

        Returns
        -------
        loadings : array [n_samples, n_components]
            The model coefficients.

        """

        loadings, components, error = self._fit(X)
        loadings = loadings[:, : self.rank_]

        return loadings

    def transform(self, Y):
        """ Apply the learned model to data Y

        Parameters
        ----------
        Y : array [n_samples, n_features]
            The data to be transformed

        Returns
        -------
        Y_transformed : array [n_samples, n_components]
            The coefficients of the Y data when projected on the
            learned basis.

        """

        if self.components_ is None:
            raise ValueError("ROSL has not been fitted to any data.")

        Y_transformed = np.dot(Y, self.components_.T)

        return Y_transformed

    def _fit(self, X):
        """ Build a model of data X

        Parameters
        ----------
        X : array [n_samples, n_features]
            The data to be modelled

        Returns
        -------
        loadings : array [n_samples, n_features]
            The subspace coefficients

        components : array [n_samples, n_features]
            The subspace basis

        E : array [n_samples, n_features]
            The error in the data model

        """
        X = self._check_array(X)
        n_samples, n_features = X.shape
        loadings = np.zeros((n_samples, n_features), dtype=np.double, order="F")
        components = np.copy(loadings)
        E = np.copy(loadings)
        s1, s2 = self.sampling
        self.rank_ = self._pyrosl(
            X,
            loadings,
            components,
            E,
            n_samples,
            n_features,
            self.rank,
            self.reg,
            self.tol,
            self.iters,
            self._mode,
            s1,
            s2,
            self.verbose,
        )

        self.components_ = components[: self.rank_]

        return loadings, components, E

    def _check_array(self, X):
        """ Sanity-checks the data and parameters.

        """
        x = np.copy(X)
        if np.isfortran(x) is False:
            print("Array must be in Fortran-order. Converting now.")
            x = np.asfortranarray(x)
        if self.sampling > x.shape:
            raise ValueError("'sampling' is greater than the dimensions of X")

        return x
