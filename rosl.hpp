/***************************************************************************

	C++ Robust Orthonormal Subspace Learning

	Author:	Tom Furnival	
	Email:	tjof2@cam.ac.uk

	Copyright (C) 2015 Tom Furnival

	This is a C++ implementation of Robust Orthonormal Subspace Learning (ROSL)
	and its fast variant, ROSL+, based on the MATLAB implementation by Xianbiao
	Shu et al. [1].

	References:
	[1] 	"Robust Orthonormal Subspace Learning: Efficient Recovery of 
			Corrupted Low-rank Matrices", (2014), Shu, X et al.
			http://dx.doi.org/10.1109/CVPR.2014.495  

***************************************************************************/

#ifndef _ROSL_HPP_
#define _ROSL_HPP_

#if defined (_WIN32)
	#if defined(rosllib_EXPORTS)
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
			// Default parameters
			method 	= 0;
			R		= 5;
			lambda 	= 0.02;
			tol 	= 1E-6;
			maxIter = 100;
			S 		= 100;
		};
		~ROSL(){
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
		void Parameters(int rankEstimate, double lambdaParameter, double tolerance, int maxiterations, int usermethod, int subsampling) {
			method 	= usermethod;
			R 		= rankEstimate;
			lambda 	= lambdaParameter;
			tol 	= tolerance;
			maxIter = maxiterations;
			S 		= subsampling;
			return;
		};
		
		void getA(double *aPy, int sz) {
			  memcpy(aPy, A.memptr(), sz*sizeof(double));
			  A.reset();
			  return;
		};
		
		void getE(double *ePy, int sz) {
			 memcpy(ePy, E.memptr(), sz*sizeof(double));
			 E.reset();
			 return;
		};
				
	private:
		// Solve full ROSL via inexact Augmented Lagrangian Multiplier method
		inline void InexactALM_ROSL(arma::mat *X);
		
		// Robust linear regression for ROSL+ via inexact Augmented Lagrangian Multiplier method		
		inline void InexactALM_RLR(arma::mat *X);
		
		// Dictionary shrinkage for full ROSL
		inline void LowRankDictionaryShrinkage(arma::mat *X);

		// User parameters
		int method, R, S, maxIter;
		double lambda, tol;

		// Basic parameters	
		int rank;
		double mu;
		
		// Armadillo matrices		
		arma::mat D, A, E, alpha, Z, Etmp, error;	
};

extern "C" {
	void pyROSL(double *xPy, double *aPy, double *ePy, int m, int n, int R, double lambda, double tol, int iter, int mode, int subsample);
}

#endif









