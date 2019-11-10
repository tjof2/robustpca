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

from pyrosl import ROSL

if __name__ == "__main__":
    # Parameters to create dataset
    n = 2000
    m = 2000
    rank = 5  # Actual rank
    p = 0.1  # Percentage of sparse errors
    seed = 12  # Random seed

    # Parameters for ROSL
    regROSL = 0.03
    estROSL = 10

    # Parameters for ROSL+
    regROSLp = 0.05
    estROSLp = 10
    samplesp = (250, 250)

    rng = np.random.RandomState(seed)

    # Basis
    U = rng.randn(n, rank)
    V = rng.randn(m, rank)
    R = np.dot(U, np.transpose(V))

    # Sparse errors
    E = -1000 + 1000 * rng.rand(n, m)
    E = rng.binomial(1, p, (n, m)) * E

    # Add the errors
    X = R + E

    # Run the sub-sampled version
    ss_rosl = ROSL(
        n_components=estROSLp,
        method="subsample",
        sampling=samplesp,
        lambda1=regROSLp,
        max_iter=1000,
        verbose=True,
    )
    _ = ss_rosl.fit_transform(X)

    # Run the full ROSL algorithm
    full_rosl = ROSL(
        n_components=estROSL,
        method="full",
        lambda1=regROSL,
        max_iter=1000,
        verbose=True,
    )
    _ = full_rosl.fit_transform(X)

    # Output some numbers
    ssmodel = ss_rosl.model_
    fullmodel = full_rosl.model_

    error1 = np.linalg.norm(R - ssmodel, "fro") / np.linalg.norm(R, "fro")
    error2 = np.linalg.norm(R - fullmodel, "fro") / np.linalg.norm(R, "fro")
    error3 = np.linalg.norm(fullmodel - ssmodel, "fro") / np.linalg.norm(
        fullmodel, "fro"
    )
    print(
        "---"
        f"Subsampled ROSL+ error: {error1:.5f}"
        f"Full ROSL error:        {error2:.5f}"
        f"ROSL/ROSL+ comparison:  {error3:.5f}"
        "---"
    )
