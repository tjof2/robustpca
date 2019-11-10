/***************************************************************************

	C++ Robust Orthonormal Subspace Learning

	Author:	Tom Furnival
	Email:	tjof2@cam.ac.uk

	Copyright (C) 2015-2019 Tom Furnival

	This is a C++ implementation of Robust Orthonormal Subspace Learning (ROSL)
	and its fast variant, ROSL+, based on the MATLAB implementation by Xianbiao
	Shu et al. [1].

	References:
	[1] 	"Robust Orthonormal Subspace Learning: Efficient Recovery of
			Corrupted Low-rank Matrices", (2014), Shu, X et al.
			http://dx.doi.org/10.1109/CVPR.2014.495

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

#ifndef _ROSL_HPP_
#define _ROSL_HPP_

#if defined (_WIN32)
	#if defined(librosl_EXPORTS)
		#define DLLEXPORT __declspec(dllexport)
	#else
		#define DLLEXPORT __declspec(dllimport)
	#endif
#else
	#define DLLEXPORT
#endif

// C++ headers
#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>
#include <cstdlib>
#include <armadillo>

class DLLEXPORT ROSL {
	public:
		ROSL(){
			// Initialize default class parameters
			method 	= 0;
			Sl 		= 100;
			Sh      = 100;
			R		= 5;
			lambda 	= 0.02;
			tol 	= 1E-6;
			maxIter = 100;
			verbose = false;
		};
		~ROSL(){
			// Clear memory
			D.reset();
			E.reset();
			A.reset();
			alpha.reset();
			Z.reset();
			Etmp.reset();
			error.reset();
		};

		// Full ROSL for data matrix X
		void runROSL(arma::mat *X);

		// Set parameters
		void Parameters(int rankEstimate, double lambdaParameter, double tolerance, int maxiterations, int usermethod, int subsamplingl, int subsamplingh, bool verb) {
			method 	= usermethod;
			Sl 		= subsamplingl;
			Sh		= subsamplingh;
			R 		= rankEstimate;
			lambda 	= lambdaParameter;
			tol 	= tolerance;
			maxIter = maxiterations;
			verbose = verb;
			return;
		};

		void getD(double *dPy, int m, int n) {
			D.resize(m, n);
			memcpy(dPy, D.memptr(), D.n_elem*sizeof(double));
			D.reset();
			return;
		};

		void getAlpha(double *alphaPy, int m, int n) {
			alpha.resize(m, n);
			memcpy(alphaPy, alpha.memptr(), alpha.n_elem*sizeof(double));
			alpha.reset();
			return;
		};

		void getE(double *ePy) {
			memcpy(ePy, E.memptr(), E.n_elem*sizeof(double));
			E.reset();
			return;
		};

		int getR() {
			return D.n_cols;
		}

	private:
		// Solve full ROSL via inexact Augmented Lagrangian Multiplier method
		void InexactALM_ROSL(arma::mat *X);

		// Robust linear regression for ROSL+ via inexact Augmented Lagrangian Multiplier method
		void InexactALM_RLR(arma::mat *X);

		// Dictionary shrinkage for full ROSL
		void LowRankDictionaryShrinkage(arma::mat *X);

		// User parameters
		int method, R, Sl, Sh, maxIter;
		double lambda, tol;
		bool verbose;

		// Basic parameters
		int rank, roslIters, rlrIters;
		double mu;

		// Armadillo matrices
		arma::mat D, A, E, alpha, Z, Etmp, error;
};

// This is the Python/C interface using ctypes
//		- Needs to be C-style for simplicity
extern "C" {
	int pyROSL(double *xPy, double *dPy, double *alphaPy, double *ePy, int m, int n, int R, double lambda, double tol, int iter, int method, int subsamplel, int subsampleh, bool verbose);
}

#endif
