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
        n = 1024
        m = 1024
        rank = 4
        p = 0.05
        seed = 123
        self.sampling = (256, 256)

        rng = np.random.RandomState(seed)

        # Basis
        U = rng.randn(n, rank)
        V = rng.randn(m, rank)
        R = U @ V.T

        # Sparse errors
        E = -1000 + 1000 * rng.rand(n, m)
        E = rng.binomial(1, p, (n, m)) * E

        # Add the errors
        self.R = R
        self.E = E
        self.X = self.R + self.E
        self.rng = rng
        self.seed = seed
        self.card = m * n

    def _verify_norms(self, A, B, tol=5e-3):
        tol *= self.card
        assert A.shape == B.shape

        for norm in ["fro", 1]:
            assert np.linalg.norm(A - B, norm) < tol

    @pytest.mark.parametrize("n_components", [5, 10])
    @pytest.mark.parametrize("lambda1", [None, 0.01, 0.1])
    def test_full(self, n_components, lambda1):
        rosl = ROSL(n_components=n_components, lambda1=lambda1, random_state=self.seed)
        _ = rosl.fit_transform(self.X)
        self._verify_norms(self.R, rosl.model_)
        self._verify_norms(self.E, rosl.residuals_)

    @pytest.mark.parametrize("n_components", [5, 10])
    @pytest.mark.parametrize("lambda1", [None, 0.01, 0.1])
    def test_subsample(self, n_components, lambda1):
        rosl = ROSL(
            n_components=n_components,
            lambda1=lambda1,
            method="subsample",
            sampling=self.sampling,
            max_iter=500,
            random_state=self.seed,
        )
        _ = rosl.fit_transform(self.X)
        self._verify_norms(self.R, rosl.model_)
        self._verify_norms(self.E, rosl.residuals_)
