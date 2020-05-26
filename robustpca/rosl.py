# -*- coding: utf-8 -*-
# Copyright 2015-2020 Tom Furnival
#
# This file is part of robustpca.
#
# robustpca is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# robustpca is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with robustpca.  If not, see <http://www.gnu.org/licenses/>.

import ctypes
import os

import numpy as np
from numpy.ctypeslib import ndpointer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted


def _cpp_rosl(
    X,
    n_components=None,
    method="full",
    sampling=None,
    lambda1=1,
    tol=1e-6,
    max_iter=1e3,
    verbose=True,
):
    if not np.isfortran(X):
        print("Array must be in Fortran-order. Converting now.")
        X = np.asfortranarray(X)

    n_samples, n_features = X.shape

    if n_components is None:
        n_components = n_features

    _available_methods = {"full": 0, "subsample": 1}

    if method not in _available_methods:
        raise NotImplementedError(
            f"'method' must be one of {_available_methods.keys()}"
        )

    if method == "subsample" and sampling is None:
        raise ValueError("'method' is set to 'subsample' but 'sampling' is not set.")

    libpath = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../src/librosl.so"
    )
    pyrosl = ctypes.cdll.LoadLibrary(libpath).pyROSL
    pyrosl.restype = ctypes.c_int
    pyrosl.argtypes = [
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

    D = np.zeros(X.shape, dtype=np.double, order="F")
    A = np.zeros(X.shape, dtype=np.double, order="F")
    E = np.zeros(X.shape, dtype=np.double, order="F")

    if sampling is not None:
        s1, s2 = sampling
    else:
        s1, s2 = n_samples, n_features

    rank_est = pyrosl(
        X,
        D,
        A,
        E,
        n_samples,
        n_features,
        n_components,
        lambda1,
        tol,
        int(max_iter),
        _available_methods[method],
        s1,
        s2,
        verbose,
    )

    return D, A, E, rank_est


class ROSL(BaseEstimator, TransformerMixin):

    """ Robust Orthonormal Subspace Learning Python wrapper.

    Robust Orthonormal Subspace Learning (ROSL) seeks to recover a low-rank matrix X
    and a sparse error matrix E from a corrupted observation Y:

        min ||X||_* + lambda ||E||_1    subject to Y = X + E

    where ||.||_* is the nuclear norm, and ||.||_1 is the l1-norm. ROSL further models
    the low-rank matrix X as spanning an orthonormal subspace D with coefficients A:

        X = D*A

    Further information can be found in the paper [Shu2014]_.

    Parameters
    ----------
    method : string, optional
        if method == 'full' (default), use full data matrix
        if method == 'subsample', use a subset of the data with a size defined
            by the 'sampling' keyword argument (ROSL+ algorithm).
    sampling : tuple (n_cols, n_rows), required if 'method' == 'subsample'
        The size of the data matrix used in the ROSL+ algorithm.
    n_components : int, optional
        Initial estimate of data dimensionality.
    lambda1 : float
        Regularization parameter on l1-norm (sparse error term).
        Default is 0.01.
    tol : float
        Stopping criterion for iterative algorithm. Default is 1e-6.
    max_iter : int
        Maximum number of iterations. Default is 500.
    verbose : bool, default True
        Show or hide the output from the algorithm.

    Attributes
    ----------
    model_ : array, [n_samples, n_features]
        The results of the ROSL decomposition.
    residuals_ : array, [n_samples, n_features]
        The error in the model.

    References
    ----------
    .. [Shu2014] X. Shu, F. Porikli and N. Ahuja, "Robust Orthonormal Subspace Learning:
                 Efficient Recovery of Corrupted Low-Rank Matrices," 2014 IEEE Conference on
                 Computer Vision and Pattern Recognition, Columbus, OH, 2014, pp. 3874-3881,
                 DOI: 10.1109/CVPR.2014.495.

    """

    def __init__(
        self,
        n_components=None,
        method="full",
        sampling=None,
        lambda1=0.01,
        tol=1e-6,
        max_iter=500,
        verbose=True,
        copy=False,
    ):
        self.n_components = n_components
        self.method = method
        self.sampling = sampling
        self.lambda1 = lambda1
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.copy = copy

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

        components : array [n_components, n_features]
            The subspace basis

        """
        X = check_array(X, copy=self.copy, dtype=[np.float64, np.float32])

        D, A, E, rank_est = _cpp_rosl(
            X,
            n_components=self.n_components,
            method=self.method,
            sampling=self.sampling,
            lambda1=self.lambda1,
            tol=self.tol,
            max_iter=self.max_iter,
            verbose=self.verbose,
        )

        self.n_components_ = rank_est
        self.loadings_ = D[:, : self.n_components_]
        self.components_ = A[: self.n_components]
        self.residuals_ = E
        self.model_ = D @ A

        return self.loadings_, self.components_

    def fit(self, X, y=None):
        """Build a model of data X.

        Parameters
        ----------
        X : array [n_samples, n_features]
            The data to be modelled.
        y : Ignored

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        self._fit(X)

        return self

    def transform(self, Y):
        """Apply the learned model to data Y.

        Parameters
        ----------
        Y : array [n_samples, n_features]
            The data to be transformed

        Returns
        -------
        Y_transformed : array [n_samples, n_features]
            The coefficients of the Y data when projected on the
            learned basis.

        """
        check_is_fitted(self, "n_components_")
        Y = check_array(Y)

        return Y @ self.components_.T
