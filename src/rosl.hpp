/***************************************************************************

  Copyright (C) 2015-2020 Tom Furnival

  This file is part of robustpca.

  robustpca is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  robustpca is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with robustpca.  If not, see <http://www.gnu.org/licenses/>.

***************************************************************************/

#ifndef _ROSL_HPP_
#define _ROSL_HPP_

#if defined(_WIN32)
#if defined(librosl_EXPORTS)
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT __declspec(dllimport)
#endif
#else
#define DLLEXPORT
#endif

#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>
#include <cstdlib>
#include <cstdint>
#include <armadillo>

class DLLEXPORT ROSL
{
public:
  ROSL()
  {
    method = 0;
    Sl = 100;
    Sh = 100;
    R = 5;
    lambda = 0.02;
    tol = 1E-6;
    maxIter = 100;
    verbose = false;
  };
  ~ROSL()
  {
    D.reset();
    E.reset();
    A.reset();
    alpha.reset();
    Z.reset();
    Etmp.reset();
    error.reset();
  };

  // Set parameters
  void Parameters(uint32_t rankEstimate, double lambdaParameter, double tolerance,
                  uint32_t maxiterations, uint32_t usermethod, uint32_t subsamplingl,
                  uint32_t subsamplingh, bool verb)
  {
    method = usermethod;
    Sl = subsamplingl;
    Sh = subsamplingh;
    R = rankEstimate;
    lambda = lambdaParameter;
    tol = tolerance;
    maxIter = maxiterations;
    verbose = verb;
    return;
  };

  // Full ROSL for data matrix X
  void runROSL(const arma::mat &X)
  {
    uint32_t m = X.n_rows;
    uint32_t n = X.n_cols;

    switch (method)
    {
    case 0: // For fully-sampled ROSL
      InexactALM_ROSL(X);
      break;
    case 1: // For sub-sampled ROSL+
      arma::uvec rowAll, colAll, rowSample, colSample;
      arma::mat Xperm;

      arma::arma_rng::set_seed_random();

      rowAll = (Sh == m) ? arma::linspace<arma::uvec>(0, m - 1, m)
                         : arma::shuffle(arma::linspace<arma::uvec>(0, m - 1, m));
      colAll = (Sl == n) ? arma::linspace<arma::uvec>(0, n - 1, n)
                         : arma::shuffle(arma::linspace<arma::uvec>(0, n - 1, n));

      rowSample = (Sh == m) ? rowAll : arma::join_vert(rowAll.subvec(0, Sh - 1), arma::sort(rowAll.subvec(Sh, m - 1)));
      colSample = (Sl == n) ? colAll : arma::join_vert(colAll.subvec(0, Sl - 1), arma::sort(colAll.subvec(Sl, n - 1)));

      Xperm = X.rows(rowSample);
      Xperm = Xperm.cols(colSample);

      // Take the columns and solve the small ROSL problem
      InexactALM_ROSL(Xperm.cols(0, Sl - 1));

      // Now take the rows and do robust linear regression
      InexactALM_RLR(Xperm.rows(0, Sh - 1));

      // Free some memory
      Xperm.reset();

      // Calculate low-rank component and error
      A = D * alpha;
      A.cols(colSample) = A;
      A.rows(rowSample) = A;
      E = X - A;

      break;
    }

    // Free some memory
    Z.reset();
    Etmp.reset();
    error.reset();
    A.reset();

    return;
  };

  void getD(double *dPy, uint32_t m, uint32_t n)
  {
    D.resize(m, n);
    memcpy(dPy, D.memptr(), D.n_elem * sizeof(double));
    D.reset();
    return;
  };

  void getAlpha(double *alphaPy, uint32_t m, uint32_t n)
  {
    alpha.resize(m, n);
    memcpy(alphaPy, alpha.memptr(), alpha.n_elem * sizeof(double));
    alpha.reset();
    return;
  };

  void getE(double *ePy)
  {
    memcpy(ePy, E.memptr(), E.n_elem * sizeof(double));
    E.reset();
    return;
  };

  uint32_t getR()
  {
    return D.n_cols;
  };

private:
  void InexactALM_ROSL(const arma::mat &X)
  {
    uint32_t m = X.n_rows;
    uint32_t n = X.n_cols;
    uint32_t precision = (uint32_t)std::abs(std::log10(tol)) + 2;

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
    A = X;
    D.zeros();
    E.zeros();
    Z.zeros();
    Etmp.zeros();

    double infNorm, froNorm, ooFroNorm;
    infNorm = arma::norm(arma::vectorise(X), "inf");
    froNorm = arma::norm(X, "fro");
    ooFroNorm = 1.0 / froNorm;

    // These are tunable parameters
    double rho, ooRho, muBar;
    mu = 10 * lambda / infNorm;
    rho = 1.5;
    ooRho = 1.0 / rho;
    muBar = mu * 1E7;

    double stopCrit;

    for (size_t i = 0; i < maxIter; i++)
    {
      // Error matrix and intensity thresholding
      Etmp = X + Z - A;
      E = arma::abs(Etmp) - lambda / mu;
      E.transform([](double val) { return (val > 0.) ? val : 0.; });
      E = E % arma::sign(Etmp);

      // Perform the shrinkage
      LowRankDictionaryShrinkage(X);

      // Update Z
      Z = ooRho * (Z + X - A - E);
      mu = (mu * rho < muBar) ? mu * rho : muBar;

      // Calculate stop criterion
      stopCrit = arma::norm(X - A - E, "fro") * ooFroNorm;
      roslIters = i + 1;

      if (stopCrit < tol)
      {
        if (verbose)
        {
          printFixed(std::cout, precision,
                     "ROSL iterations:  ", i + 1,
                     "\nEstimated rank:   ", D.n_cols,
                     "\nFinal error:      ", stopCrit);
        }
        return;
      }
    }

    printFixed(std::cerr, precision,
               "WARNING: ROSL did not converge in ", roslIters, " iterations",
               "\nEstimated rank:   ", D.n_cols,
               "\nFinal error:      ", stopCrit);

    return;
  };

  void InexactALM_RLR(const arma::mat &X)
  {
    uint32_t m = X.n_rows;
    uint32_t n = X.n_cols;
    uint32_t precision = (uint32_t)std::abs(std::log10(tol)) + 2;

    // Initialize A, Z, E, Etmp
    A.set_size(m, n);
    Z.set_size(m, n);
    E.set_size(m, n);
    Etmp.set_size(m, n);

    // Set all other matrices
    A = X;
    E.zeros();
    Z.zeros();
    Etmp.zeros();

    double infNorm, froNorm, ooFroNorm;
    infNorm = arma::norm(arma::vectorise(X), "inf");
    froNorm = arma::norm(X, "fro");
    ooFroNorm = 1.0 / froNorm;

    // These are tunable parameters
    double rho, ooRho, muBar;
    mu = 10 * 5E-2 / infNorm;
    rho = 1.5;
    ooRho = 1.0 / rho;
    muBar = mu * 1E7;

    double stopCrit;

    for (size_t i = 0; i < maxIter; i++)
    {
      // Error matrix and intensity thresholding
      Etmp = X + Z - A;
      E = arma::abs(Etmp) - 1 / mu;
      E.transform([](double val) { return (val > 0.) ? val : 0.; });
      E = E % arma::sign(Etmp);

      // SVD variables
      arma::mat Usvd, Vsvd;
      arma::vec Ssvd;
      arma::uvec Sshort;
      uint32_t SV;

      // Given D and A...
      arma::svd_econ(Usvd, Ssvd, Vsvd, D.rows(0, Sh - 1));
      Sshort = arma::find(Ssvd > 0.);
      SV = Sshort.n_elem;
      alpha = Vsvd.cols(0, SV - 1) * arma::diagmat(1. / Ssvd.subvec(0, SV - 1)) *
              arma::trans(Usvd.cols(0, SV - 1)) * (X + Z - E);
      A = (D.rows(0, Sh - 1)) * alpha;

      // Update Z
      Z = (Z + X - A - E) * ooRho;
      mu = (mu * rho < muBar) ? mu * rho : muBar;

      // Calculate stop criterion
      stopCrit = arma::norm(X - A - E, "fro") * ooFroNorm;
      rlrIters = i + 1;

      // Exit if stop criteria is met
      if (stopCrit < tol)
      {
        if (verbose)
        {
          printFixed(std::cout, precision,
                     "RLR iterations:   ", i + 1,
                     "\nEstimated rank:   ", D.n_cols,
                     "\nFinal error:      ", stopCrit);
        }
        return;
      }
    }

    printFixed(std::cerr, precision,
               "WARNING: RLR did not converge in ", rlrIters, " iterations",
               "\nEstimated rank:   ", D.n_cols,
               "\nFinal error:      ", stopCrit);

    return;
  };

  void LowRankDictionaryShrinkage(const arma::mat &X)
  {
    rank = D.n_cols;

    double dNorm, alphaNormThresh, ooMu;
    ooMu = 1.0 / mu;

    arma::vec alphaNorm = arma::zeros<arma::vec>(rank);
    arma::uvec alphaIdxs;

    for (size_t i = 0; i < rank; i++) // Loop over columns of D
    {
      // Compute error and new D(:,i)
      D.col(i).zeros();
      error = ((X + Z - E) - (D * alpha));
      D.col(i) = error * arma::trans(alpha.row(i));
      dNorm = arma::norm(D.col(i));

      if (dNorm > 0.) // Shrinkage with Gram-Schmidt on D
      {
        for (size_t j = 0; j < i; j++)
        {
          D.col(i) = D.col(i) - D.col(j) * (arma::trans(D.col(j)) * D.col(i));
        }

        // Normalize
        D.col(i) /= arma::norm(D.col(i));

        // Compute alpha(i,:)
        alpha.row(i) = arma::trans(D.col(i)) * error;

        // Magnitude thresholding
        alphaNorm(i) = arma::norm(alpha.row(i));
        alphaNormThresh = std::max(alphaNorm(i) - ooMu, 0.0);
        alpha.row(i) *= alphaNormThresh / alphaNorm(i);
        alphaNorm(i) = alphaNormThresh;
      }
      else
      {
        alpha.row(i).zeros();
        alphaNorm(i) = 0.;
      }
    }

    // Delete the zero bases and update A
    alphaIdxs = arma::find(alphaNorm != 0.);
    D = D.cols(alphaIdxs);
    alpha = alpha.rows(alphaIdxs);
    A = D * alpha;

    return;
  };

  uint32_t method, R, Sl, Sh, maxIter;
  double lambda, tol;
  bool verbose;

  uint32_t rank, roslIters, rlrIters;
  double mu;

  arma::mat D, A, E, alpha, Z, Etmp, error;

  template <typename Arg, typename... Args>
  void print(std::ostream &out, Arg &&arg, Args &&... args)
  {
    out << std::forward<Arg>(arg);
    using expander = int[];
    (void)expander{0, (void(out << std::forward<Args>(args)), 0)...};
    out << std::endl;
  }

  template <typename Arg, typename... Args>
  void printFixed(std::ostream &out, const uint32_t precision, Arg &&arg, Args &&... args)
  {
    print(out, std::fixed, std::setprecision(precision), arg, args...);
  }
};

extern "C"
{
  // This is the Python/C interface using ctypes
  // (needs to be C-style for simplicity)
  int pyROSL(double *xPy, double *dPy, double *alphaPy, double *ePy, int m, int n,
             int R, double lambda, double tol, int iter, int method,
             int subsamplel, int subsampleh, bool verbose)
  {

    // Create class instance
    ROSL *pyrosl = new ROSL();

    // First pass the parameters (the easy bit!)
    pyrosl->Parameters(R, lambda, tol, iter, method, subsamplel, subsampleh, verbose);

    // Copy the image sequence into arma::mat
    // This is the dangerous bit - we want to avoid copying, so set
    // up the Armadillo data matrix to DIRECTLY read from auxiliary
    // memory, but be careful, this is also writable! Remember also
    // that Armadillo stores in column-major order.
    arma::mat X(xPy, m, n, false, false);

    // Time ROSL
    auto t0 = std::chrono::high_resolution_clock::now();

    // Run ROSL
    pyrosl->runROSL(X);

    auto t1 = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    if (verbose)
    {
      std::cout << "Total time: " << std::setprecision(5)
                << elapsed / 1E6 << " seconds" << std::endl;
    }

    // Get the estimated rank
    uint32_t rankEst = pyrosl->getR();

    // Now copy the data back to return to Python
    pyrosl->getD(dPy, m, n);
    pyrosl->getAlpha(alphaPy, m, n);
    pyrosl->getE(ePy);

    // Free memory
    delete pyrosl;

    // Return the rank
    return rankEst;
  }
}

#endif
