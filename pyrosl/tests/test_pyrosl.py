# -*- coding: utf-8 -*-
# Copyright 2019 Tom Furnival
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

import numpy as np
import pytest

from pyrosl import ROSL

DEFAULT_TOL = 5e-3


class TestROSL:
    def setup_method(self, method):
        # Parameters to create dataset
        n = 2000
        m = 2000
        rank = 5
        p = 0.1
        seed = 12

        # Parameters for ROSL

        # Parameters for ROSL+
        reg_s = 0.05
        est_s = 10
        sampling = (250, 250)

        rng = np.random.RandomState(seed)

        # Basis
        U = rng.randn(n, rank)
        V = rng.randn(m, rank)
        R = np.dot(U, np.transpose(V))

        # Sparse errors
        E = -1000 + 1000 * rng.rand(n, m)
        E = rng.binomial(1, p, (n, m)) * E

        # Add the errors
        self.R = R
        self.E = E
        self.X = self.R + self.E
        self.rng = rng
        self.card = m * n

    def _verify_norms(self, A, B, tol=DEFAULT_TOL):
        assert A.shape == B.shape

        for norm in ["fro", 1]:
            normX = np.linalg.norm(A - B, norm) / (self.card)
            assert normX < tol

            normE = np.linalg.norm(A - B, norm) / (self.card)
            assert normE < tol

    def test_full(self):
        rosl = ROSL(n_components=10, lambda1=0.03)
        _ = rosl.fit_transform(X)
        self._verify_norms(self.R, rosl.model_)
        self._verify_norms(self.E, rosl.errors_)

    def test_subsample(self):
        rosl = ROSL(
            n_components=est_s,
            method="subsample",
            sampling=sampling,
            lambda1=reg_s,
            max_iter=1000,
        )
        _ = rosl.fit_transform(X)
        self._verify_norms(self.R, rosl.model_)
        self._verify_norms(self.E, rosl.errors_)
