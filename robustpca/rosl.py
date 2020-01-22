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

import ctypes
import os
import numpy as np

from numpy.ctypeslib import ndpointer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_array
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

    libpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "librosl.so.0.2")
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
    alpha = np.zeros(X.shape, dtype=np.double, order="F")
    E = np.zeros(X.shape, dtype=np.double, order="F")

    s1, s2 = sampling
    _ = pyrosl(
        X,
        D,
        alpha,
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

    return D, alpha, E


class ROSL(BaseEstimator, TransformerMixin):

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

    n_components : int, optional
        Initial estimate of data dimensionality.

    lambda1 : float, optional
        Regularization parameter on l1-norm (sparse error term).

    tol : float, optional
        Stopping criterion for iterative algorithm.

    max_iter : int, optional
        Maximum number of iterations.

    verbose : bool, optional
        Show or hide the output from the algorithm.

    use_cpp : bool, optional
        Use C++ algorithm instead of Python.

    Attributes
    ----------
    model_ : array, [n_samples, n_features]
        The results of the ROSL decomposition.

    residuals_ : array, [n_samples, n_features]
        The error in the model.

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
        use_cpp=False,
        random_state=None,
    ):
        self.n_components = n_components
        self.method = method
        self.sampling = sampling
        self.lambda1 = lambda1
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.use_cpp = use_cpp
        self.random_state = random_state

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
        X = check_array(X, dtype=float)

        if self.use_cpp:
            D, alpha, E = _cpp_rosl(
                X,
                n_components=self.n_components,
                method=self.method,
                sampling=self.sampling,
                lambda1=self.lambda1,
                tol=self.tol,
                max_iter=self.max_iter,
                verbose=self.verbose,
            )

        self.n_components_ = n_components
        self.loadings_ = D[:, : self.n_components_]
        self.components_ = alpha[: self.n_components]
        self.residuals_ = E
        self.model_ = D.dot(A)

        return self.loadings_, self.components_

    def fit(self, X, y=None):
        """ Build a model of data X

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

    def fit_transform(self, X, y=None):
        """ Build a model of data X and apply it to data X

        Parameters
        ----------
        X : array [n_samples, n_features]
            The data to be modelled.
        y : Ignored

        Returns
        -------
        loadings : array [n_samples, n_features]
            The model coefficients.

        """
        loadings, components = self._fit(X)
        loadings = loadings[:, : self.n_components_]

        return loadings

    def transform(self, Y):
        """ Apply the learned model to data Y.

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

        return np.dot(Y, self.components_.T)
