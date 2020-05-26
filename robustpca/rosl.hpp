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

#include <iostream>
#include <iomanip>
#include <random>
#include <cstdlib>
#include <cstdint>
#include <armadillo>

namespace rosl
{
  class ROSL
  {
  public:
    ROSL(const double lambda,
         const double tol,
         const uint32_t maxRank,
         const uint32_t maxIter) : lambda(lambda),
                                   tol(tol),
                                   maxRank(maxRank),
                                   maxIter(maxIter)
    {
      sampleL = 0;
      sampleH = 0;
      precision = static_cast<uint32_t>(std::abs(std::log10(tol)) + 3); // Will print 3 decimal places
    };

    ROSL(const double lambda,
         const double tol,
         const uint32_t maxRank,
         const uint32_t maxIter,
         const uint32_t sampleL,
         const uint32_t sampleH) : lambda(lambda),
                                   tol(tol),
                                   maxRank(maxRank),
                                   maxIter(maxIter),
                                   sampleL(sampleL),
                                   sampleH(sampleH)
    {
      precision = static_cast<uint32_t>(std::abs(std::log10(tol)) + 3); // Will print 3 decimal places
    };

    ~ROSL()
    {
      Z.reset();
      D.reset();
      B.reset();
      E_.reset();
      C.reset();
    };

    void InexactALM_ROSL(const arma::mat &X, arma::mat &A, arma::mat &E)
    {
      uint32_t m = X.n_rows;
      uint32_t n = X.n_cols;

      A.set_size(m, n); // Initialize
      E.set_size(m, n);
      Z.set_size(m, n);
      D.set_size(m, maxRank);
      B.set_size(maxRank, n);
      E_.set_size(m, n);
      C.set_size(m, n);

      B.randn();                 // Initialize B randomly
      B /= arma::norm(B, "fro"); // Normalize B

      A = X;     // First guess of A is the input data
      D.zeros(); // Set all other matrices to zero
      E.zeros();
      Z.zeros();
      E_.zeros();

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
        E_ = X + Z - A;
        E = arma::abs(E_) - lambda / mu;
        E.transform([](double val) { return std::max(val, 0.0); });
        E %= arma::sign(E_);

        LowRankDictionaryShrinkage(X, A, E);

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

    void InexactALM_RLR(const arma::mat &X, arma::mat &A, arma::mat &E)
    {
      uint32_t m = X.n_rows;
      uint32_t n = X.n_cols;

      A.set_size(m, n); // Initialize
      E.set_size(m, n);
      Z.set_size(m, n);
      E_.set_size(m, n);

      A = X;     // First guess of A is the input data
      E.zeros(); // Set all other matrices to zero
      Z.zeros();
      E_.zeros();

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
        E_ = X + Z - A;
        E = arma::abs(E_) - 1.0 / mu;
        E.transform([](double val) { return std::max(val, 0.0); });
        E %= arma::sign(E_);

        // Given D and A, apply SVD to update B
        arma::svd_econ(U_, S_, V_, D.rows(0, sampleH - 1));
        S_nz = arma::find(S_ > 0.);
        sV = std::max(static_cast<int>(S_nz.n_elem), 1) - 1;

        B = V_.cols(0, sV) * arma::diagmat(1.0 / S_.subvec(0, sV)) * arma::trans(U_.cols(0, sV)) * (X + Z - E);

        A = (D.rows(0, sampleH - 1)) * B; // Update A
        Z = (Z + X - A - E) * ooRho;      // Update Z
        mu = std::min(mu * rho, muBar);   // Update mu

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

    arma::mat D, B; // Basis and dictionary are public

  private:
    double lambda, tol;
    uint32_t maxRank, maxIter, sampleL, sampleH;

    uint32_t precision, rank;
    double mu;

    arma::mat Z, E_, C;

    void LowRankDictionaryShrinkage(const arma::mat &X, arma::mat &A, arma::mat &E)
    {
      rank = D.n_cols;

      double dNorm, bNormThresh, ooMu;
      ooMu = 1.0 / mu;

      arma::vec bNorm = arma::zeros<arma::vec>(rank);
      arma::uvec bIdxs;

      for (size_t i = 0; i < rank; i++) // Loop over columns of D
      {
        // Compute error and new D(:,i)
        D.col(i).zeros();
        C = ((X + Z - E) - (D * B));
        D.col(i) = C * arma::trans(B.row(i));
        dNorm = arma::norm(D.col(i));

        if (dNorm > 0.) // Shrinkage with Gram-Schmidt on D
        {
          for (size_t j = 0; j < i; j++)
          {
            D.col(i) = D.col(i) - D.col(j) * (arma::trans(D.col(j)) * D.col(i));
          }

          D.col(i) /= arma::norm(D.col(i)); // Normalize

          B.row(i) = arma::trans(D.col(i)) * C; // Compute B(i,:)

          // Magnitude thresholding
          bNorm(i) = arma::norm(B.row(i));
          bNormThresh = std::max(bNorm(i) - ooMu, 0.0);
          B.row(i) *= bNormThresh / bNorm(i);
          bNorm(i) = bNormThresh;
        }
        else
        {
          B.row(i).zeros();
          bNorm(i) = 0.;
        }
      }

      // Delete the zero bases and update A
      bIdxs = arma::find(bNorm != 0.);
      if (bIdxs.n_elem < 1)
      {
        D = D.cols(0, 0);
        B = B.rows(0, 0);
        print(std::cerr, "WARNING: all bases are zero. ",
              "Consider increasing sampling rate or ",
              "regularization parameter.");
      }
      else
      {
        D = D.cols(bIdxs);
        B = B.rows(bIdxs);
      }
      A = D * B;

      return;
    };

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

} // namespace rosl

uint32_t rosl_lrs(const arma::mat &X,
                  arma::mat &A,
                  arma::mat &E,
                  const double lambda,
                  const double tol,
                  const bool subsample,
                  const double sampleLFrac,
                  const double sampleHFrac,
                  const uint32_t maxRank,
                  const uint32_t maxIter,
                  const int randomSeed)
{
  uint32_t rankEstimate;
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

  if (subsample) // For sub-sampled ROSL+
  {
    uint32_t minSamples = 1;
    uint32_t sampleH = std::max(minSamples, std::min(n, static_cast<uint32_t>(std::floor(n * sampleLFrac))));
    uint32_t sampleL = std::max(minSamples, std::min(m, static_cast<uint32_t>(std::floor(m * sampleHFrac))));

    rosl::ROSL *est = new rosl::ROSL(lambda, tol, maxRank, maxIter, sampleL, sampleH);

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

    est->InexactALM_ROSL(Xperm.cols(0, sampleL - 1), A, E); // Take the columns and solve the small ROSL problem
    est->InexactALM_RLR(Xperm.rows(0, sampleH - 1), A, E);  // Now take the rows and do robust linear regression

    Xperm.reset(); // Free some memory

    // Calculate low-rank component and error
    A = est->D * est->B;
    A.cols(colSample) = A;
    A.rows(rowSample) = A;
    E = X - A;

    rankEstimate = static_cast<uint32_t>(est->D.n_cols); // Final rank estimate
    delete est;                                          // Free memory
  }
  else
  { // For fully-sampled ROSL
    rosl::ROSL *est = new rosl::ROSL(lambda, tol, maxRank, maxIter);
    est->InexactALM_ROSL(X, A, E);

    rankEstimate = static_cast<uint32_t>(est->D.n_cols); // Final rank estimate
    delete est;                                          // Free memory
  }

  return rankEstimate;
}

uint32_t rosl_all(const arma::mat &X,
                  arma::mat &A,
                  arma::mat &E,
                  arma::mat &D,
                  arma::mat &B,
                  const double lambda,
                  const double tol,
                  const bool subsample,
                  const double sampleLFrac,
                  const double sampleHFrac,
                  const uint32_t maxRank,
                  const uint32_t maxIter,
                  const int randomSeed)
{
  uint32_t rankEstimate;
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

  if (subsample) // For sub-sampled ROSL+
  {
    uint32_t minSamples = 1;
    uint32_t sampleH = std::max(minSamples, std::min(n, static_cast<uint32_t>(std::floor(n * sampleLFrac))));
    uint32_t sampleL = std::max(minSamples, std::min(m, static_cast<uint32_t>(std::floor(m * sampleHFrac))));

    rosl::ROSL *est = new rosl::ROSL(lambda, tol, maxRank, maxIter, sampleL, sampleH);

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

    est->InexactALM_ROSL(Xperm.cols(0, sampleL - 1), A, E); // Take the columns and solve the small ROSL problem
    est->InexactALM_RLR(Xperm.rows(0, sampleH - 1), A, E);  // Now take the rows and do robust linear regression

    Xperm.reset(); // Free some memory

    // Calculate low-rank component and error
    A = est->D * est->B;
    A.cols(colSample) = A;
    A.rows(rowSample) = A;
    E = X - A;

    D.copy_size(est->D); // Copy basis and dictionary
    B.copy_size(est->B);
    D = est->D;
    B = est->B;

    rankEstimate = static_cast<uint32_t>(est->D.n_cols); // Final rank estimate
    delete est;
  }
  else
  { // For fully-sampled ROSL
    rosl::ROSL *est = new rosl::ROSL(lambda, tol, maxRank, maxIter);
    est->InexactALM_ROSL(X, A, E);

    D.copy_size(est->D); // Copy basis and dictionary
    B.copy_size(est->B);
    D = est->D;
    B = est->B;

    rankEstimate = static_cast<uint32_t>(est->D.n_cols); // Final rank estimate
    delete est;                                          // Free memory
  }

  return rankEstimate;
}

#endif
