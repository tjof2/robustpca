/***************************************************************************
Copyright (C) 2015-2019 Tom Furnival

This file is part of RobustPCA.

RobustPCA is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RobustPCA is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with RobustPCA.  If not, see <http://www.gnu.org/licenses/>.
***************************************************************************/

#include "rosl.hpp"

void ROSL::runROSL(arma::mat *X) {
  int m = (*X).n_rows;
  int n = (*X).n_cols;

  switch (method) {
  case 0:
    // For fully-sampled ROSL
    InexactALM_ROSL(X);
    break;
  case 1:
    // For sub-sampled ROSL+
    arma::uvec rowall, colall;
    arma::arma_rng::set_seed_random();
    rowall = (Sh == m) ? arma::linspace<arma::uvec>(0, m - 1, m)
                       : arma::shuffle(arma::linspace<arma::uvec>(0, m - 1, m));
    colall = (Sl == n) ? arma::linspace<arma::uvec>(0, n - 1, n)
                       : arma::shuffle(arma::linspace<arma::uvec>(0, n - 1, n));

    arma::uvec rowsample, colsample;
    rowsample = (Sh == m)
                    ? rowall
                    : arma::join_vert(rowall.subvec(0, Sh - 1),
                                      arma::sort(rowall.subvec(Sh, m - 1)));
    colsample = (Sl == n)
                    ? colall
                    : arma::join_vert(colall.subvec(0, Sl - 1),
                                      arma::sort(colall.subvec(Sl, n - 1)));

    arma::mat Xperm;
    Xperm = (*X).rows(rowsample);
    Xperm = Xperm.cols(colsample);

    // Take the columns and solve the small ROSL problem
    arma::mat XpermTmp;
    XpermTmp = Xperm.cols(0, Sl - 1);
    InexactALM_ROSL(&XpermTmp);

    // Free some memory
    XpermTmp.set_size(Sh, Sl);

    // Now take the rows and do robust linear regression
    XpermTmp = Xperm.rows(0, Sh - 1);
    InexactALM_RLR(&XpermTmp);

    // Free some memory
    Xperm.reset();
    XpermTmp.reset();

    // Calculate low-rank component
    A = D * alpha;

    // Permute back
    A.cols(colsample) = A;
    A.rows(rowsample) = A;

    // Calculate error
    E = *X - A;
    break;
  }

  // Free some memory
  Z.reset();
  Etmp.reset();
  error.reset();
  A.reset();

  return;
};

void ROSL::InexactALM_ROSL(arma::mat *X) {
  int m = (*X).n_rows;
  int n = (*X).n_cols;
  int precision = (int)std::abs(std::log10(tol)) + 2;

  // Initialize A, Z, E, Etmp and error
  A.set_size(m, n);
  Z.set_size(m, n);
  E.set_size(m, n);
  Etmp.set_size(m, n);
  alpha.set_size(R, n);
  D.set_size(m, R);
  error.set_size(m, n);

  // Initialize alpha randomly
  arma::arma_rng::set_seed_random();
  alpha.randu();

  // Set all other matrices
  A = *X;
  D.zeros();
  E.zeros();
  Z.zeros();
  Etmp.zeros();

  double infnorm, fronorm;
  infnorm = arma::norm(arma::vectorise(*X), "inf");
  fronorm = arma::norm(*X, "fro");

  // These are tunable parameters
  double rho, mubar;
  mu = 10 * lambda / infnorm;
  rho = 1.5;
  mubar = mu * 1E7;

  double stopcrit;

  for (int i = 0; i < maxIter; i++) {
    // Error matrix and intensity thresholding
    Etmp = *X + Z - A;
    E = arma::abs(Etmp) - lambda / mu;
    E.transform([](double val) { return (val > 0.) ? val : 0.; });
    E = E % arma::sign(Etmp);

    // Perform the shrinkage
    LowRankDictionaryShrinkage(X);

    // Update Z
    Z = (Z + *X - A - E) / rho;
    mu = (mu * rho < mubar) ? mu * rho : mubar;

    // Calculate stop criterion
    stopcrit = arma::norm(*X - A - E, "fro") / fronorm;
    roslIters = i + 1;

    // Exit if stop criteria is met
    if (stopcrit < tol) {
      if (verbose) {
        std::cout << "   ROSL iterations: " << i + 1 << std::endl;
        std::cout << "    Estimated rank: " << D.n_cols << std::endl;
        std::cout << "       Final error: " << std::fixed
                  << std::setprecision(precision) << stopcrit << std::endl;
      }
      return;
    }
  }

  std::cout << "   WARNING: ROSL did not converge in " << roslIters
            << " iterations" << std::endl;
  std::cout << "            Estimated rank:  " << D.n_cols << std::endl;
  std::cout << "               Final error: " << std::fixed
            << std::setprecision(precision) << stopcrit << std::endl;

  return;
};

void ROSL::InexactALM_RLR(arma::mat *X) {
  int m = (*X).n_rows;
  int n = (*X).n_cols;
  int precision = (int)std::abs(std::log10(tol)) + 2;

  // Initialize A, Z, E, Etmp
  A.set_size(m, n);
  Z.set_size(m, n);
  E.set_size(m, n);
  Etmp.set_size(m, n);

  // Set all other matrices
  A = *X;
  E.zeros();
  Z.zeros();
  Etmp.zeros();

  double infnorm, fronorm;
  infnorm = arma::norm(arma::vectorise(*X), "inf");
  fronorm = arma::norm(*X, "fro");

  // These are tunable parameters
  double rho, mubar;
  mu = 10 * 5E-2 / infnorm;
  rho = 1.5;
  mubar = mu * 1E7;

  double stopcrit;

  for (int i = 0; i < maxIter; i++) {
    // Error matrix and intensity thresholding
    Etmp = *X + Z - A;
    E = arma::abs(Etmp) - 1 / mu;
    E.transform([](double val) { return (val > 0.) ? val : 0.; });
    E = E % arma::sign(Etmp);

    // SVD variables
    arma::mat Usvd, Vsvd;
    arma::vec Ssvd;
    arma::uvec Sshort;
    int SV;

    // Given D and A...
    arma::svd_econ(Usvd, Ssvd, Vsvd, D.rows(0, Sh - 1));
    Sshort = arma::find(Ssvd > 0.);
    SV = Sshort.n_elem;
    alpha = Vsvd.cols(0, SV - 1) * arma::diagmat(1. / Ssvd.subvec(0, SV - 1)) *
            arma::trans(Usvd.cols(0, SV - 1)) * (*X + Z - E);
    A = (D.rows(0, Sh - 1)) * alpha;

    // Update Z
    Z = (Z + *X - A - E) / rho;
    mu = (mu * rho < mubar) ? mu * rho : mubar;

    // Calculate stop criterion
    stopcrit = arma::norm(*X - A - E, "fro") / fronorm;
    rlrIters = i + 1;

    // Exit if stop criteria is met
    if (stopcrit < tol) {
      if (verbose) {
        std::cout << "    RLR iterations: " << i + 1 << std::endl;
        std::cout << "    Estimated rank: " << D.n_cols << std::endl;
        std::cout << "       Final error: " << std::fixed
                  << std::setprecision(precision) << stopcrit << std::endl;
      }
      return;
    }
  }

  std::cout << "   WARNING: RLR did not converge in " << rlrIters
            << " iterations" << std::endl;
  std::cout << "            Estimated rank:  " << D.n_cols << std::endl;
  std::cout << "               Final error: " << std::fixed
            << std::setprecision(precision) << stopcrit << std::endl;

  return;
};

void ROSL::LowRankDictionaryShrinkage(arma::mat *X) {
  // Get current rank estimate
  rank = D.n_cols;

  // Thresholding
  double alphanormthresh;
  arma::vec alphanorm(rank);
  alphanorm.zeros();
  arma::uvec alphaindices;

  // Norms
  double dnorm;

  // Loop over columns of D
  for (int i = 0; i < rank; i++) {
    // Compute error and new D(:,i)
    D.col(i).zeros();
    error = ((*X + Z - E) - (D * alpha));
    D.col(i) = error * arma::trans(alpha.row(i));
    dnorm = arma::norm(D.col(i));

    // Shrinkage
    if (dnorm > 0.) {
      // Gram-Schmidt on D
      for (int j = 0; j < i; j++) {
        D.col(i) = D.col(i) - D.col(j) * (arma::trans(D.col(j)) * D.col(i));
      }

      // Normalize
      D.col(i) /= arma::norm(D.col(i));

      // Compute alpha(i,:)
      alpha.row(i) = arma::trans(D.col(i)) * error;

      // Magnitude thresholding
      alphanorm(i) = arma::norm(alpha.row(i));
      alphanormthresh =
          (alphanorm(i) - 1 / mu > 0.) ? alphanorm(i) - 1 / mu : 0.;
      alpha.row(i) *= alphanormthresh / alphanorm(i);
      alphanorm(i) = alphanormthresh;
    } else {
      alpha.row(i).zeros();
      alphanorm(i) = 0.;
    }
  }

  // Delete the zero bases
  alphaindices = arma::find(alphanorm != 0.);
  D = D.cols(alphaindices);
  alpha = alpha.rows(alphaindices);

  // Update A
  A = D * alpha;

  return;
};

// This is the Python/C interface using ctypes (needs to be C-style for simplicity)
int pyROSL(double *xPy, double *dPy, double *alphaPy, double *ePy, int m, int n,
           int R, double lambda, double tol, int iter, int method,
           int subsamplel, int subsampleh, bool verbose) {

  // Create class instance
  ROSL *pyrosl = new ROSL();

  // First pass the parameters (the easy bit!)
  pyrosl->Parameters(R, lambda, tol, iter, method, subsamplel, subsampleh,
                     verbose);

  /////////////////////
  //                 //
  // !!! WARNING !!! //
  //                 //
  /////////////////////

  // This is the dangerous bit - we want to avoid copying, so set up the
  // Armadillo data matrix to DIRECTLY read from auxiliary memory,
  // but be careful, this is also writable!!!
  // Remember also that Armadillo stores in column-major order
  arma::mat X(xPy, m, n, false, false);

  // Time ROSL
  auto t0 = std::chrono::high_resolution_clock::now();

  // Run ROSL
  pyrosl->runROSL(&X);

  auto t1 = std::chrono::high_resolution_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

  if (verbose) {
    std::cout << "Total time: " << std::setprecision(5)
              << elapsed / 1E6 << " seconds" << std::endl;
  }

  // Get the estimated rank
  int rankEst = pyrosl->getR();

  // Now copy the data back to return to Python
  pyrosl->getD(dPy, m, n);
  pyrosl->getAlpha(alphaPy, m, n);
  pyrosl->getE(ePy);

  // Free memory
  delete pyrosl;

  // Return the rank
  return rankEst;
}
