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

from scipy.linalg import qr

from robustpca import ROSL


class TestROSL:
    def setup_method(self, method):
        # Parameters to create dataset
        self.n_samples = 256
        self.n_features = 256
        self.n_components = 5
        self.seed = 123
        self.card = self.n_samples * self.n_features
        self.subsampling = (0.25, 0.25)

        self.rng = np.random.RandomState(self.seed)

        self.U = self.rng.randn(self.n_samples, self.n_components)
        self.U, _ = qr(self.U, mode="economic", check_finite=False)
        self.V = self.rng.randn(self.n_features, self.n_components)
        self.A = self.U @ self.V.T
        np.divide(self.A, max(1.0, np.linalg.norm(self.A)), out=self.A)

        self.E = 5 * self.rng.binomial(1, 0.05, (self.n_samples, self.n_features))
        self.X = self.A + self.E

    def _check_properties(self, rosl, tol=5e-3):
        tol *= self.card

        # Verify all the matrix dimensions are consistent
        assert rosl.low_rank_.shape == self.A.shape
        assert rosl.error_.shape == self.A.shape

        # Check features/samples dimensions are consistent
        assert rosl.loadings_.shape[0] == self.U.shape[0]
        assert rosl.components_.shape[1] == self.V.T.shape[1]

        # Check the L2 distance between the recovered components
        assert np.linalg.norm(rosl.low_rank_ - self.A, "fro") < tol
        assert np.linalg.norm(rosl.error_ - self.E, "fro") < tol

        # Check the L1 distance between the recovered components
        assert np.linalg.norm(rosl.low_rank_ - self.A, 1) < tol
        assert np.linalg.norm(rosl.error_ - self.E, 1) < tol

        # Residual should be approx ~N(0, sigma), so
        # check that the mean is ~= 0
        assert np.abs((self.X - rosl.low_rank_ - rosl.error_).mean()) < tol

    @pytest.mark.parametrize("n_components", [5, 10])
    @pytest.mark.parametrize("lambda1", [None, 0.01, 0.1])
    @pytest.mark.parametrize("max_iter", [250, 500])
    @pytest.mark.parametrize("tol", [1e-6, 1e-7])
    def test_full(self, n_components, lambda1, max_iter, tol, capfd):
        rosl = ROSL(
            n_components=n_components,
            lambda1=lambda1,
            max_iter=max_iter,
            tol=tol,
            random_seed=self.seed,
        )
        _ = rosl.fit_transform(self.X)

        if np.any(np.isnan(rosl.low_rank_)):
            captured = capfd.readouterr()
            assert "WARNING: all bases are zero." in captured.err
            assert np.all(np.isnan(rosl.low_rank_))
            assert np.all(np.isnan(rosl.error_))
        else:
            self._check_properties(rosl)

    @pytest.mark.parametrize("n_components", [5, 10])
    @pytest.mark.parametrize("lambda1", [None, 0.01, 0.1])
    @pytest.mark.parametrize("subsampling", [0.25, (0.25, 0.33)])
    @pytest.mark.parametrize("max_iter", [250, 500])
    @pytest.mark.parametrize("tol", [1e-6, 1e-7])
    def test_subsample(self, n_components, lambda1, subsampling, max_iter, tol, capfd):
        rosl = ROSL(
            n_components=n_components,
            lambda1=lambda1,
            subsampling=subsampling,
            max_iter=max_iter,
            tol=tol,
            random_seed=self.seed,
        )
        _ = rosl.fit_transform(self.X)

        if np.any(np.isnan(rosl.low_rank_)):
            captured = capfd.readouterr()
            assert "WARNING: all bases are zero." in captured.err
            assert np.all(np.isnan(rosl.low_rank_))
            assert np.all(np.isnan(rosl.error_))
        else:
            self._check_properties(rosl)

    def test_defaults(self, capfd):
        rosl = ROSL()
        _ = rosl.fit_transform(self.X)

        if np.any(np.isnan(rosl.low_rank_)):
            captured = capfd.readouterr()
            assert "WARNING: all bases are zero." in captured.err
            assert np.all(np.isnan(rosl.low_rank_))
            assert np.all(np.isnan(rosl.error_))
        else:
            self._check_properties(rosl)

    def test_errors(self):
        with pytest.raises(ValueError, match="instead of a tuple of len=2"):
            rosl = ROSL(subsampling=(0.5, 0.5, 0.5))
            _ = rosl.fit_transform(self.X)

        with pytest.raises(ValueError, match="floats between 0 and 1"):
            rosl = ROSL(subsampling=(0.5, -0.5))
            _ = rosl.fit_transform(self.X)
