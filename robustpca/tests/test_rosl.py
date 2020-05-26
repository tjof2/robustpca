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
import pytest

from robustpca import ROSL


class TestROSL:
    def setup_method(self, method):
        # Parameters to create dataset
        n = 1000
        m = 1000
        rank = 5
        p = 0.1
        seed = 12

        # Parameters for ROSL

        # Parameters for ROSL+
        self.reg_s = 0.05
        self.est_s = 10
        self.sampling = (250, 250)

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

    def _verify_norms(self, A, B, tol=1e-3):
        tol *= self.card
        assert A.shape == B.shape

        for norm in ["fro", 1]:
            assert np.linalg.norm(A - B, norm) < tol

    def test_full(self):
        rosl = ROSL(n_components=10, lambda1=0.03)
        _ = rosl.fit_transform(self.X)
        self._verify_norms(self.R, rosl.model_)
        self._verify_norms(self.E, rosl.residuals_)

    def test_subsample(self):
        rosl = ROSL(
            n_components=self.est_s,
            method="subsample",
            sampling=self.sampling,
            lambda1=self.reg_s,
            max_iter=1000,
        )
        _ = rosl.fit_transform(self.X)
        self._verify_norms(self.R, rosl.model_)
        self._verify_norms(self.E, rosl.residuals_)
