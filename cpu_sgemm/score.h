#ifndef SCORE_H
#define SCORE_H

#include <cmath>
#include <emmintrin.h>

void cmpResults(float *A,float *B, float *C, float *D, unsigned int N, std::string a, std::string b){
	float diff = std::abs(C[0] - D[0]);
	float maxA = std::abs(C[0]);
	float maxB = std::abs(D[0]);

	for(unsigned int i = 0; i <N *N; i++){
		if(std::abs(C[i] - D[i]) > diff) diff = std::abs(C[i] - D[i]);
		if(std::abs(C[i]) > maxA) maxA = std::abs(A[i]);
		if(std::abs(D[i]) > maxB) maxB = std::abs(B[i]);
	}
	diff/=maxA*maxB;
	std::cout<<"maximum difference between "<<a << " and "<<b  <<" is " << diff <<std::endl;
}

void sgemm_base(const float *A, const float *B, float *&C, unsigned int N){
	for(unsigned int i = 0; i < N ;i++){
		for(unsigned int j = 0; j < N ;j++){
			for(unsigned int k = 0; k < N ;k++){
				C[i * N + j] += A[ i * N + k ] * B[ k * N + j ];
			}
		}
	}
}


void sgemm_regC(const float *A, const float *B, float *&C, unsigned int N){
	for(unsigned int i = 0; i < N ;i++){
		for(unsigned int j = 0; j < N ;j++){
			register float rC = 0.0f;
			for(unsigned int k = 0; k < N ;k++){
				rC += A[ i * N + k ] * B[ k * N + j ];
			}
			C[i * N + j] = rC;
		}
	}
}

void sgemm_cache(const float *A, const float *B, float *&C, unsigned int N){
	for(unsigned int i = 0; i < N ; i++){
		for(unsigned int k = 0; k < N ;k++){
			register float rA = A[ i * N + k ];
			for(unsigned int j = 0 ; j < N ; j++){
				C[i * N + j] += rA * B[ k * N + j ];
			}
		}
	}
}


void sgemm_cache2(const float *A, const float *B, float *&C, unsigned int N){
	for(unsigned int i = 0; i < N ; i++){
		for(unsigned int k = 0; k < N ;k+=4){
			register unsigned int iA = i * N + k;
			register float rA = A[ iA ];
			register float rA1 = A[ iA + 1 ];
			register float rA2 = A[ iA + 2 ];
			register float rA3 = A[ iA + 3 ];

			register float rC = 0.0f;
			for(unsigned int j = 0 ; j < N ; j+=4){
				register unsigned int iB = k * N + j;
				register unsigned int iiB = iB + N;
				register unsigned int iiiB = iiB + N;
				register unsigned int iiiiB = iiiB + N;

				register unsigned int iC = i * N + j;

				C[iC] += rA * B[ iB ] + rA1 * B[ iiB ] + rA2 * B[ iiiB ] + rA3 * B[ iiiiB ];
				C[iC + 1] += rA * B[ iB + 1 ] + rA1 * B[ iiB + 1 ] + rA2 * B[ iiiB + 1 ] + rA3 * B[ iiiiB + 1 ];
				C[iC + 2] += rA * B[ iB + 2 ] + rA1 * B[ iiB + 2 ] + rA2 * B[ iiiB + 2 ] + rA3 * B[ iiiiB + 2 ];
				C[iC + 3] += rA * B[ iB + 3 ] + rA1 * B[ iiB + 3 ] + rA2 * B[ iiiB + 3 ] + rA3 * B[ iiiiB + 3 ];
			}
		}
	}
}

void sgemm_block(const float *A, const float *B, float *&C, unsigned int N){
	unsigned int Bz = 32;
	for(unsigned int i = 0; i < N; i+=Bz){
		for(unsigned int j = 0; j < N; j+=Bz){
			for(unsigned int k = 0; k < N; k+=Bz){
				//Blocked Matrix
				for(unsigned int ii = i; ii < i + Bz; ii++){
					for(unsigned int jj = j; jj < j + Bz; jj++){
						register float rC = C[ii * N + jj];
						for(unsigned int kk = k; kk < k + Bz; kk++){
								rC+= A[ii * N + kk] * B[kk*N + jj];
						}
						C[ii * N + jj] = rC;
					}
				}

			}
		}
	}
}

void sgemm_score_main(float *A, float *B, float *&C, float *&D, unsigned int N){
	//1
	start_clock();
	sgemm_base(A,B,C,N);
	stop_clock();
	double sgemm_base = secf();

	//2
	start_clock();
	sgemm_regC(A,B,D,N);
	stop_clock();
	double sgemm_regC = secf();
	cmpResults(A,B,C,D,N,"sgemm_base","sgemm_regC");
	zeros(D,N);

	//3
	start_clock();
	sgemm_cache2(A,B,D,N);
	stop_clock();
	double sgemm_cache = secf();
	cmpResults(A,B,C,D,N,"sgemm_base","sgemm_cache");
	zeros(D,N);

	start_clock();
	sgemm_block(A,B,D,N);
	stop_clock();
	double sgemm_block = secf();
	cmpResults(A,B,C,D,N,"sgemm_base","sgemm_block");

	double GFLOPS = (double)(N*N*N*2);
	std::cout<< "Elapsed time for sgemm_base: " << sgemm_base << " seconds\n";
	std::cout<< "Elapsed time for sgemm_regC: " << sgemm_regC << " seconds\n";
	std::cout<< "Elapsed time for sgemm_cache: " << sgemm_cache << " seconds\n";
	std::cout<< "Elapsed time for sgemm_block: " << sgemm_block << " seconds\n";

	std::cout << "GFLOPS for sgemm_base: " << ((double)(GFLOPS/sgemm_base))/1000000000 << "\n";
	std::cout << "GFLOPS for sgemm_regC: " << ((double)(GFLOPS/sgemm_regC))/1000000000 << "\n";
	std::cout << "GFLOPS for sgemm_cache: " << ((double)(GFLOPS/sgemm_cache))/1000000000 << "\n";
	std::cout << "GFLOPS for sgemm_block: " << ((double)(GFLOPS/sgemm_block))/1000000000 << "\n";

}

#endif
