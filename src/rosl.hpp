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

#include <iostream>
#include <iomanip>
#include <random>
#include <cstdlib>
#include <cstdint>
#include <armadillo>

class DLLEXPORT ROSL
{
public:
  ROSL(const double lambda,
       const double tol,
       const uint32_t method,
       const uint32_t maxRank,
       const uint32_t maxIter,
       const uint32_t sampleL,
       const uint32_t sampleH,
       const int randomSeed) : lambda(lambda),
                               tol(tol),
                               method(method),
                               maxRank(maxRank),
                               maxIter(maxIter),
                               sampleL(sampleL),
                               sampleH(sampleH),
                               randomSeed(randomSeed)
  {
    precision = static_cast<uint32_t>(std::abs(std::log10(tol)) + 3); // Will print 3 decimal places
  };

  ~ROSL()
  {
    D.reset();
    E.reset();
    A.reset();
    Z.reset();
    alpha.reset();
    tempE.reset();
    error.reset();
  };

  // Full ROSL for data matrix X
  void runROSL(const arma::mat &X)
  {
    uint32_t m = X.n_rows;
    uint32_t n = X.n_cols;

    if (randomSeed < 0)
    {
      arma::arma_rng::set_seed_random();
    }
    else
    {
      arma::arma_rng::set_seed(randomSeed);
    }

    switch (method)
    {
    case 0: // For fully-sampled ROSL
      InexactALM_ROSL(X);
      break;
    case 1: // For sub-sampled ROSL+
      arma::uvec rowAll, colAll, rowSample, colSample;
      arma::mat Xperm;

      rowAll = (sampleH == m) ? arma::linspace<arma::uvec>(0, m - 1, m)
                              : arma::shuffle(arma::linspace<arma::uvec>(0, m - 1, m));
      colAll = (sampleL == n) ? arma::linspace<arma::uvec>(0, n - 1, n)
                              : arma::shuffle(arma::linspace<arma::uvec>(0, n - 1, n));

      rowSample = (sampleH == m) ? rowAll : arma::join_vert(rowAll.subvec(0, sampleH - 1), arma::sort(rowAll.subvec(sampleH, m - 1)));
      colSample = (sampleL == n) ? colAll : arma::join_vert(colAll.subvec(0, sampleL - 1), arma::sort(colAll.subvec(sampleL, n - 1)));

      Xperm = X.rows(rowSample);
      Xperm = Xperm.cols(colSample);

      InexactALM_ROSL(Xperm.cols(0, sampleL - 1)); // Take the columns and solve the small ROSL problem
      InexactALM_RLR(Xperm.rows(0, sampleH - 1));  // Now take the rows and do robust linear regression

      Xperm.reset(); // Free some memory

      // Calculate low-rank component and error
      A = D * alpha;
      A.cols(colSample) = A;
      A.rows(rowSample) = A;
      E = X - A;

      break;
    }

    // Free some memory
    Z.reset();
    A.reset();
    tempE.reset();
    error.reset();

    return;
  };

  void getD(double *dPy, uint32_t m, uint32_t n)
  {
    D.resize(m, n);
    memcpy(dPy, D.memptr(), D.n_elem * sizeof(double));
    D.reset();
    return;
  };

  void getA(double *aPy, uint32_t m, uint32_t n)
  {
    alpha.resize(m, n);
    memcpy(aPy, alpha.memptr(), alpha.n_elem * sizeof(double));
    alpha.reset();
    return;
  };

  void getE(double *ePy)
  {
    memcpy(ePy, E.memptr(), E.n_elem * sizeof(double));
    E.reset();
    return;
  };

  uint32_t getRank()
  {
    return D.n_cols;
  };

  void InexactALM_ROSL(const arma::mat &X)
  {
    uint32_t m = X.n_rows;
    uint32_t n = X.n_cols;

    A.set_size(m, n); // Initialize
    Z.set_size(m, n);
    E.set_size(m, n);
    D.set_size(m, maxRank);
    tempE.set_size(m, n);
    alpha.set_size(maxRank, n);
    error.set_size(m, n);

    alpha.randu(); // Initialize alpha randomly
    A = X;         // First guess of A is the input data
    D.zeros();     // Set all other matrices to zero
    E.zeros();
    Z.zeros();
    tempE.zeros();

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
      // Error matrix thresholding
      tempE = X + Z - A;
      E = arma::abs(tempE) - lambda / mu;
      E.transform([](double val) { return std::max(val, 0.0); });
      E %= arma::sign(tempE);

      LowRankDictionaryShrinkage(X);

      Z = ooRho * (Z + X - A - E);    // Update Z
      mu = std::min(mu * rho, muBar); // Update mu

      stopCrit = arma::norm(X - A - E, "fro") * ooFroNorm;
      if (stopCrit < tol)
      {
        printFixed(std::cout, precision,
                   "ROSL iterations:  ", i + 1,
                   "\nEstimated rank:   ", D.n_cols,
                   "\nFinal error:      ", stopCrit);
        return;
      }
    }

    printFixed(std::cerr, precision,
               "WARNING: ROSL did not converge in ", maxIter, " iterations",
               "\nEstimated rank:   ", D.n_cols,
               "\nFinal error:      ", stopCrit);

    return;
  };

  void InexactALM_RLR(const arma::mat &X)
  {
    uint32_t m = X.n_rows;
    uint32_t n = X.n_cols;

    A.set_size(m, n); // Initialize
    Z.set_size(m, n);
    E.set_size(m, n);
    tempE.set_size(m, n);

    A = X;     // First guess of A is the input data
    E.zeros(); // Set all other matrices to zero
    Z.zeros();
    tempE.zeros();

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

    arma::mat U_, V_; // SVD variables
    arma::vec S_;
    arma::uvec S_nz;
    int sV;

    for (size_t i = 0; i < maxIter; i++)
    {
      // Error matrix thresholding
      tempE = X + Z - A;
      E = arma::abs(tempE) - 1.0 / mu;
      E.transform([](double val) { return std::max(val, 0.0); });
      E %= arma::sign(tempE);

      // Given D and A, apply SVD to update alpha
      arma::svd_econ(U_, S_, V_, D.rows(0, sampleH - 1));
      S_nz = arma::find(S_ > 0.);
      sV = std::max(static_cast<int>(S_nz.n_elem), 1) - 1;

      // std::cout << D.rows(0, sampleH - 1).n_elem << std::endl
      //           << S_.n_elem << std::endl
      //           << S_nz.n_elem << std::endl
      //           << sV << std::endl;

      alpha = V_.cols(0, sV) * arma::diagmat(1.0 / S_.subvec(0, sV)) * arma::trans(U_.cols(0, sV)) * (X + Z - E);

      A = (D.rows(0, sampleH - 1)) * alpha; // Update A
      Z = (Z + X - A - E) * ooRho;          // Update Z
      mu = std::min(mu * rho, muBar);       // Update mu

      stopCrit = arma::norm(X - A - E, "fro") * ooFroNorm;
      if (stopCrit < tol)
      {
        printFixed(std::cout, precision,
                   "RLR iterations:   ", i + 1,
                   "\nEstimated rank:   ", D.n_cols,
                   "\nFinal error:      ", stopCrit);
        return;
      }
    }

    printFixed(std::cerr, precision,
               "WARNING: RLR did not converge in ", maxIter, " iterations",
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

        D.col(i) /= arma::norm(D.col(i)); // Normalize

        alpha.row(i) = arma::trans(D.col(i)) * error; // Compute alpha(i,:)

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
    if (alphaIdxs.n_elem < 1)
    {
      D = D.cols(0, 0);
      alpha = alpha.rows(0, 0);
      print(std::cerr, "WARNING: all bases are zero. ",
            "Consider increasing sampling rate or ",
            "regularization parameter.");
    }
    else
    {
      D = D.cols(alphaIdxs);
      alpha = alpha.rows(alphaIdxs);
    }
    A = D * alpha;

    return;
  };

private:
  double lambda, tol;
  uint32_t method, maxRank, maxIter, sampleL, sampleH;
  int randomSeed;

  uint32_t precision, rank;
  double mu;

  arma::mat D, A, E, alpha, Z, tempE, error;

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
  uint32_t pyROSL(double *xPy, double *dPy, double *aPy, double *ePy,
                  double lambda, double tol,
                  uint32_t m, uint32_t n, uint32_t maxRank,
                  uint32_t iter, uint32_t method,
                  uint32_t sampleL, uint32_t sampleH,
                  int randomSeed)
  {
    // Copy the image sequence into arma::mat
    // This is the dangerous bit - we want to avoid copying, so set
    // up the Armadillo data matrix to DIRECTLY read from auxiliary
    // memory, but be careful, this is also writable! Remember also
    // that Armadillo stores in column-major order.
    arma::mat X(xPy, m, n, false, false);

    ROSL *est = new ROSL(lambda, tol, method, maxRank, iter, sampleL, sampleH, randomSeed);
    est->runROSL(X);

    // Fetch data to return to Python
    uint32_t rankEstimate = est->getRank();
    est->getD(dPy, m, n);
    est->getA(aPy, m, n);
    est->getE(ePy);

    delete est; // Free memory

    return rankEstimate; // Return the rank
  }
}

#endif
