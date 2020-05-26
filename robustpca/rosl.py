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

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from ._rosl import rosl_all


class ROSL(BaseEstimator, TransformerMixin):
    """Robust Orthonormal Subspace Learning (ROSL).

    Robust Orthonormal Subspace Learning seeks to recover a low-rank matrix X
    and a sparse error matrix E from a corrupted observation Y:

        min ||X||_* + lambda ||E||_1    subject to Y = X + E

    where ||.||_* is the nuclear norm, and ||.||_1 is the l1-norm. ROSL further models
    the low-rank matrix X as spanning an orthonormal subspace D with coefficients A:

        X = D*A

    Further information can be found in the paper [Shu2014]_.

    Parameters
    ----------
    subsampling : None or float or tuple(float, float)
        * If None, use full data matrix
        * If float, use a random fraction of the data (ROSL+ algorithm)
        * If tuple of floats, use a random fraction of the columns and rows (ROSL+ algorithm)
    n_components : int, optional
        Initial estimate of data dimensionality.
    lambda1 : float
        Regularization parameter on l1-norm (sparse error term).
        Default is 0.01.
    tol : float
        Stopping criterion for iterative algorithm. Default is 1e-6.
    max_iter : int
        Maximum number of iterations. Default is 500.
    copy : bool, default False
        If True, fit on a copy of the data.
    random_seed : None or int
        Random seed used to sample the data and initialize the starting point.
        Default is None.

    Attributes
    ----------
    low_rank_ : array, [n_samples, n_features]
        The results of the ROSL decomposition.
    error_ : array, [n_samples, n_features]
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
        subsampling=None,
        lambda1=0.01,
        max_iter=500,
        tol=1e-6,
        copy=False,
        random_seed=None,
    ):
        self.n_components = n_components
        self.subsampling = subsampling
        self.lambda1 = lambda1
        self.max_iter = max_iter
        self.tol = tol
        self.copy = copy
        self.random_seed = random_seed

    def _fit(self, X):
        """Build a model of data X.

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
        if not np.isfortran(X):
            X = np.asfortranarray(X)

        self.n_samples, self.n_features = X.shape

        if self.lambda1 is None:
            self.lambda1_ = 1.0 / np.sqrt(self.n_features)
        else:
            self.lambda1_ = self.lambda1

        if self.n_components is None:
            self.n_components = self.n_features

        self.max_iter = int(self.max_iter)

        if self.subsampling is None:
            sampling = False
            s1, s2 = (1.0, 1.0)
        elif isinstance(self.subsampling, tuple):
            if len(self.subsampling) != 2:
                raise ValueError(
                    "Invalid subsampling parameter: got tuple of len="
                    f"{len(self.subsampling)} instead of a tuple of len=2."
                )
            sampling = True
            s1, s2 = self.subsampling
        else:
            sampling = True
            s1 = self.subsampling
            s2 = self.subsampling

        if s1 > 1.0 or s1 < 0.0 or s2 > 1.0 or s2 < 0.0:
            raise ValueError(
                f"Invalid subsampling parameter: got {self.subsampling} "
                "instead of a float or pair of floats between 0 and 1."
            )

        if self.random_seed is None:
            # Negative integer used to seed randomly in C++
            self.random_seed_ = -1
        else:
            self.random_seed_ = self.random_seed

        A, E, D, B, rank_est = rosl_all(
            X,
            self.lambda1_,
            self.tol,
            sampling,
            self.n_components,
            self.max_iter,
            s1,
            s2,
            self.random_seed_,
        )

        self.n_components_ = int(rank_est)
        self.loadings_ = D[:, : self.n_components_]
        self.components_ = B[: self.n_components_]
        self.error_ = E
        self.low_rank_ = A

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
