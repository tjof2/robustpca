// C++ headers
#include <chrono>

// ROSL header
#include "rosl.hpp"

// Main program
int main(int argc, char** argv) {

	// Create low-rank matrix
	/*int L1 = 144*144;
	int L2 = 3000;
	int K = 5;
	arma::arma_rng::set_seed(101115);
	arma::mat yl = arma::randu(L1,K);
	arma::mat yr = arma::randu(K,L2);
	arma::mat Y = yl * yr;
	
	// Rescale and add noise
	Y = (Y - Y.min())/(Y.max() - Y.min());	
	Y += 0.1*arma::randn(L1,L2);*/
	
	// Read in low-rank matrix
	arma::Mat<unsigned char> D;
	D.load("../../../Dataset.arma.mat",arma::arma_binary);
	arma::mat Y = arma::conv_to<arma::mat>::from(D);
	D.reset();
	arma::inplace_trans(Y);
	
	// Subtract the mean
	arma::mat meanY = arma::repmat(arma::mean(Y), Y.n_rows, 1);
	Y -= meanY;	
	
	// Create class instance
	ROSL *example = new ROSL();	
		
	// Run ROSL
	auto timerS1 = std::chrono::steady_clock::now();
	example->Parameters(3, 0.02, 0.00001, 50, 1, 300);
	example->runROSL(&Y);
	auto timerE1 = std::chrono::steady_clock::now();
	auto elapsed1 = std::chrono::duration_cast<std::chrono::microseconds>(timerE1 - timerS1);
	std::cout<<std::endl<<"Total time: "<<std::setprecision(5)<<(elapsed1.count()/1E6)<<" seconds"<<std::endl<<std::endl;	
	
	// Run ROSL+
	auto timerS2 = std::chrono::steady_clock::now();
	example->Parameters(3, 0.015, 0.00001, 50, 0, 300);
	example->runROSL(&Y);
	auto timerE2 = std::chrono::steady_clock::now();
	auto elapsed2 = std::chrono::duration_cast<std::chrono::microseconds>(timerE2 - timerS2);
	std::cout<<std::endl<<"Total time: "<<std::setprecision(5)<<(elapsed2.count()/1E6)<<" seconds"<<std::endl<<std::endl;	
	
	// Free memory	
	delete example;
	
}
