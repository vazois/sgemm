#ifndef SCORE_H
#define SCORE_H

#include <cmath>
#include <mmintrin.h>
#include <xmmintrin.h>
//#include <emmintrin.h>
//#include <pmmintrin.h>
//#include <xmmintrin.h>
//#include <smmintrin.h>
#include <stdint.h>
#include <cblas.h>
#include "../time/Time.h"

void cmpResults(double *A,double *B, double *C, double *D, uint64_t N, std::string a, std::string b){
	double diff = std::abs(C[0] - D[0]);
	double maxA = std::abs(C[0]);
	double maxB = std::abs(D[0]);

	double eps = 1.e-6;
	for(uint64_t i = 0; i <N *N; i++){
		if(std::abs(C[i] - D[i]) > diff) diff = std::abs(C[i] - D[i]);
		if(std::abs(C[i]) > maxA) maxA = std::abs(A[i]);
		if(std::abs(D[i]) > maxB) maxB = std::abs(B[i]);
	}
	diff/=maxA*maxB;
	if(diff < eps) std::cout << "(PASS) "<< a << " = " << b << std::endl;
	else std::cout << "(ERROR) "<< a << " = " << b << " (" << diff <<")"<< std::endl;
}

void dgemm_base(const double *A, const double *B, double *&C, uint64_t N){
	for(uint64_t i = 0; i < N ;i++){
		for(uint64_t j = 0; j < N ;j++){
			for(uint64_t k = 0; k < N ;k++){
				C[i * N + j] += A[ i * N + k ] * B[ k * N + j ];
			}
		}
	}
}

void dgemm_reg_reuse(const double *A, const double *B, double *&C, uint64_t N){
	for(uint64_t i = 0; i < N ;i++){
		for(uint64_t j = 0; j < N ;j++){
			register double rC = 0.0f;
			for(uint64_t k = 0; k < N ;k++){
				rC += A[ i * N + k ] * B[ k * N + j ];
			}
			C[i * N + j] = rC;
		}
	}
}

void dgemm_reg_blocking(const double *A, const double *B, double *&C, uint64_t N){

	for(unsigned int i = 0; i < N ;i+=2){
		for(unsigned int j = 0; j < N ;j+=2){
			register uint64_t iC = i * N + j; register uint64_t iiC = iC + N;
			register double rC00 = 0.0f;
			register double rC01 = 0.0f;
			register double rC10 = 0.0f;
			register double rC11 = 0.0f;

			for(unsigned int k = 0; k < N ;k+=2){
				register uint64_t iA = i * N + k; register uint64_t iiA = iA + N;
				register uint64_t iB = k * N + j; register uint64_t iiB = iB + N;

				register double rA00 = A[ iA ]; register double rA01 = A[ iA + 1 ];
				register double rA10 = A[ iiA ]; register double rA11 = A[ iiA + 1 ];
				register double rB00 = B[ iB ]; register double rB01 = B[ iB + 1 ];
				register double rB10 = B[ iiB ]; register double rB11 = B[ iiB + 1 ];

				rC00 += rA00 * rB00; rC00+=rA01 * rB10;

				rC01 += rA00 * rB01; rC01+=rA01 * rB11;
				rC10 += rA10 * rB00; rC10+=rA11 * rB10;
				rC11 += rA10 * rB01; rC11+=rA11 * rB11;
			}
			C[ iC ] = rC00;
			C[ iC + 1 ] = rC01;
			C[ iiC ] = rC10;
			C[ iiC + 1 ] = rC11;
		}
	}
}

void dgemm_reorder(const double *A, const double *B, double *&C, uint64_t N){
	for(uint64_t i = 0; i < N ; i++){
		for(uint64_t k = 0; k < N ;k+=4){
			register uint64_t iA = i * N + k;
			register double rA = A[ iA ];
			register double rA1 = A[ iA + 1 ];
			register double rA2 = A[ iA + 2 ];
			register double rA3 = A[ iA + 3 ];

			register double rC = 0.0f;
			for(uint64_t j = 0 ; j < N ; j+=4){
				register uint64_t iB = k * N + j;
				register uint64_t iiB = iB + N;
				register uint64_t iiiB = iiB + N;
				register uint64_t iiiiB = iiiB + N;

				register uint64_t iC = i * N + j;

				C[iC] += rA * B[ iB ] + rA1 * B[ iiB ] + rA2 * B[ iiiB ] + rA3 * B[ iiiiB ];
				C[iC + 1] += rA * B[ iB + 1 ] + rA1 * B[ iiB + 1 ] + rA2 * B[ iiiB + 1 ] + rA3 * B[ iiiiB + 1 ];
				C[iC + 2] += rA * B[ iB + 2 ] + rA1 * B[ iiB + 2 ] + rA2 * B[ iiiB + 2 ] + rA3 * B[ iiiiB + 2 ];
				C[iC + 3] += rA * B[ iB + 3 ] + rA1 * B[ iiB + 3 ] + rA2 * B[ iiiB + 3 ] + rA3 * B[ iiiiB + 3 ];
			}
		}
	}
}

void dgemm_reorder_sse2(const double *A, const double *B, double *&C, uint64_t N){
	for(uint64_t i = 0; i < N ; i++){
		for(uint64_t k = 0; k < N ;k+=4){
			register uint64_t iA = i * N + k;
			__m128d vA1 = _mm_load_pd1(&A[iA]);//rA
			__m128d vA2 = _mm_load_pd1(&A[iA+1]);//rA1
			__m128d vA3 = _mm_load_pd1(&A[iA+2]);//rA2
			__m128d vA4 = _mm_load_pd1(&A[iA+3]);//rA3

			for(uint64_t j = 0 ; j < N ; j+=4){
				register uint64_t iB = k * N + j;
				register uint64_t iiB = iB + N;
				register uint64_t iiiB = iiB + N;
				register uint64_t iiiiB = iiiB + N;

				register uint64_t iC = i * N + j;


				//1
				__m128d vB1 = _mm_load_pd(&B[iB]);//rB,rB1
				__m128d vB2 = _mm_load_pd(&B[iB+2]);//rB2,rB3
				__m128d vC1 = _mm_load_pd(&C[iC]);//rC,rC1
				__m128d vC2 = _mm_load_pd(&C[iC+2]);//rC2,rC3
				vC1 = _mm_add_pd(vC1,_mm_mul_pd(vA1,vB1));
				vC2 = _mm_add_pd(vC2,_mm_mul_pd(vA1,vB2));

				//2
				vB1 = _mm_load_pd(&B[iiB]);
				vB2 = _mm_load_pd(&B[iiB+2]);
				vC1 = _mm_add_pd(vC1,_mm_mul_pd(vA2,vB1));
				vC2 = _mm_add_pd(vC2,_mm_mul_pd(vA2,vB2));

				//3
				vB1 = _mm_load_pd(&B[iiiB]);
				vB2 = _mm_load_pd(&B[iiiB+2]);
				vC1 = _mm_add_pd(vC1,_mm_mul_pd(vA3,vB1));
				vC2 = _mm_add_pd(vC2,_mm_mul_pd(vA3,vB2));

				//4
				vB1 = _mm_load_pd(&B[iiiiB]);
				vB2 = _mm_load_pd(&B[iiiiB+2]);
				vC1 = _mm_add_pd(vC1,_mm_mul_pd(vA4,vB1));
				vC2 = _mm_add_pd(vC2,_mm_mul_pd(vA4,vB2));

				_mm_store_pd(&C[iC],vC1);
				_mm_store_pd(&C[iC+2],vC2);
			}
		}
	}
}

void dgemm_reorder_sse(const double *A, const double *B, double *&C, uint64_t N){
	for(uint64_t i = 0; i < N ; i+=2){
		for(uint64_t k = 0; k < N ;k+=4){
			register uint64_t iA = i * N + k;
			register uint64_t iiA = iA + N;
			__m128d vA1 = _mm_load_pd1(&A[iA]);//rA
			__m128d vA2 = _mm_load_pd1(&A[iA+1]);//rA1
			__m128d vA3 = _mm_load_pd1(&A[iA+2]);//rA2
			__m128d vA4 = _mm_load_pd1(&A[iA+3]);//rA3

			__m128d vA11 = _mm_load_pd1(&A[iiA]);//rA
			__m128d vA12 = _mm_load_pd1(&A[iiA+1]);//rA1
			__m128d vA13 = _mm_load_pd1(&A[iiA+2]);//rA2
			__m128d vA14 = _mm_load_pd1(&A[iiA+3]);//rA3

			for(uint64_t j = 0 ; j < N ; j+=4){
				register uint64_t iB = k * N + j;
				register uint64_t iiB = iB + N;
				register uint64_t iiiB = iiB + N;
				register uint64_t iiiiB = iiiB + N;

				register uint64_t iC = i * N + j;
				register uint64_t iiC = iC + N;


				//1
				__m128d vB1 = _mm_load_pd(&B[iB]);//rB,rB1
				__m128d vB2 = _mm_load_pd(&B[iB+2]);//rB2,rB3
				__m128d vC1 = _mm_load_pd(&C[iC]);//rC,rC1
				__m128d vC2 = _mm_load_pd(&C[iC+2]);//rC2,rC3
				__m128d vC3 = _mm_load_pd(&C[iiC]);//rC,rC1
				__m128d vC4 = _mm_load_pd(&C[iiC+2]);//rC2,rC3


				vC1 = _mm_add_pd(vC1,_mm_mul_pd(vA1,vB1));
				vC2 = _mm_add_pd(vC2,_mm_mul_pd(vA1,vB2));
				vC3 = _mm_add_pd(vC3,_mm_mul_pd(vA11,vB1));
				vC4 = _mm_add_pd(vC4,_mm_mul_pd(vA11,vB2));

				//2
				vB1 = _mm_load_pd(&B[iiB]);
				vB2 = _mm_load_pd(&B[iiB+2]);
				vC1 = _mm_add_pd(vC1,_mm_mul_pd(vA2,vB1));
				vC2 = _mm_add_pd(vC2,_mm_mul_pd(vA2,vB2));
				vC3 = _mm_add_pd(vC3,_mm_mul_pd(vA12,vB1));
				vC4 = _mm_add_pd(vC4,_mm_mul_pd(vA12,vB2));

				//3
				vB1 = _mm_load_pd(&B[iiiB]);
				vB2 = _mm_load_pd(&B[iiiB+2]);
				vC1 = _mm_add_pd(vC1,_mm_mul_pd(vA3,vB1));
				vC2 = _mm_add_pd(vC2,_mm_mul_pd(vA3,vB2));
				vC3 = _mm_add_pd(vC3,_mm_mul_pd(vA13,vB1));
				vC4 = _mm_add_pd(vC4,_mm_mul_pd(vA13,vB2));

				//4
				vB1 = _mm_load_pd(&B[iiiiB]);
				vB2 = _mm_load_pd(&B[iiiiB+2]);
				vC1 = _mm_add_pd(vC1,_mm_mul_pd(vA4,vB1));
				vC2 = _mm_add_pd(vC2,_mm_mul_pd(vA4,vB2));
				vC3 = _mm_add_pd(vC3,_mm_mul_pd(vA14,vB1));
				vC4 = _mm_add_pd(vC4,_mm_mul_pd(vA14,vB2));

				_mm_store_pd(&C[iC],vC1);
				_mm_store_pd(&C[iC+2],vC2);
				_mm_store_pd(&C[iiC],vC3);
				_mm_store_pd(&C[iiC+2],vC4);
			}
		}
	}
}

void dgemm_block2(const double *A, const double *B, double *&C, uint64_t N){
	uint64_t Bx = 4;
	uint64_t By = 512;
	uint64_t Bz = 4;

	double sA[Bx * Bz];
	for(uint64_t i = 0; i < N; i+=Bx){
		for(uint64_t j = 0; j < N; j+=By){
			for(uint64_t k = 0; k < N; k+=Bz){
				//Blocked Matrix

				for(uint64_t ii = 0; ii < Bx; ii++){
					for(uint64_t kk = 0; kk < Bz; kk++){
						sA[ii * Bz + kk]=A[(i+ii) * N + k + kk];
					}
				}
				for(uint64_t ii = 0; ii < Bx; ii++){
					for(uint64_t kk = 0; kk < Bz; kk++){
						register double rA = sA[ii * Bz + kk];
						for(uint64_t jj = j; jj < j + By; jj++){
							C[(ii + i) * N + jj]+= rA * B[(kk + k) * N + jj];
						}
					}
				}

			}
		}
	}
}

void dgemm_block(const double *A, const double *B, double *&C, uint64_t N){
	uint64_t Bx = 4;
	uint64_t By = 1024;
	uint64_t Bz = 4;
	for(uint64_t i = 0; i < N; i+=Bx){
		for(uint64_t k = 0; k < N; k+=Bz){
			for(uint64_t j = 0; j < N; j+=By){
				//Blocked Matrix

				for(uint64_t ii = i; ii < i + Bx ; ii+=2){
					for(uint64_t kk = k; kk < k + Bz ;kk+=4){
						register uint64_t iA = ii * N + kk;
						register uint64_t iiA = iA + N;
						__m128d vA1 = _mm_load_pd1(&A[iA]);//rA
						__m128d vA2 = _mm_load_pd1(&A[iA+1]);//rA1
						__m128d vA3 = _mm_load_pd1(&A[iA+2]);//rA2
						__m128d vA4 = _mm_load_pd1(&A[iA+3]);//rA3

						__m128d vA11 = _mm_load_pd1(&A[iiA]);//rA
						__m128d vA12 = _mm_load_pd1(&A[iiA+1]);//rA1
						__m128d vA13 = _mm_load_pd1(&A[iiA+2]);//rA2
						__m128d vA14 = _mm_load_pd1(&A[iiA+3]);//rA3

						for(uint64_t jj = j ; jj < j + By ; jj+=4){
							register uint64_t iB = kk * N + jj;
							register uint64_t iiB = iB + N;
							register uint64_t iiiB = iiB + N;
							register uint64_t iiiiB = iiiB + N;

							register uint64_t iC = ii * N + jj;
							register uint64_t iiC = iC + N;

							__m128d vB1 = _mm_load_pd(&B[iB]);//rB,rB1
							__m128d vB2 = _mm_load_pd(&B[iB+2]);//rB2,rB3
							__m128d vC1 = _mm_load_pd(&C[iC]);//rC,rC1
							__m128d vC2 = _mm_load_pd(&C[iC+2]);//rC2,rC3
							__m128d vC3 = _mm_load_pd(&C[iiC]);//rC,rC1
							__m128d vC4 = _mm_load_pd(&C[iiC+2]);//rC2,rC3

							vC1 = _mm_add_pd(vC1,_mm_mul_pd(vA1,vB1));
							vC2 = _mm_add_pd(vC2,_mm_mul_pd(vA1,vB2));
							vC3 = _mm_add_pd(vC3,_mm_mul_pd(vA11,vB1));
							vC4 = _mm_add_pd(vC4,_mm_mul_pd(vA11,vB2));

							vB1 = _mm_load_pd(&B[iiB]);
							vB2 = _mm_load_pd(&B[iiB+2]);
							vC1 = _mm_add_pd(vC1,_mm_mul_pd(vA2,vB1));
							vC2 = _mm_add_pd(vC2,_mm_mul_pd(vA2,vB2));
							vC3 = _mm_add_pd(vC3,_mm_mul_pd(vA12,vB1));
							vC4 = _mm_add_pd(vC4,_mm_mul_pd(vA12,vB2));

							vB1 = _mm_load_pd(&B[iiiB]);
							vB2 = _mm_load_pd(&B[iiiB+2]);
							vC1 = _mm_add_pd(vC1,_mm_mul_pd(vA3,vB1));
							vC2 = _mm_add_pd(vC2,_mm_mul_pd(vA3,vB2));
							vC3 = _mm_add_pd(vC3,_mm_mul_pd(vA13,vB1));
							vC4 = _mm_add_pd(vC4,_mm_mul_pd(vA13,vB2));

							vB1 = _mm_load_pd(&B[iiiiB]);
							vB2 = _mm_load_pd(&B[iiiiB+2]);
							vC1 = _mm_add_pd(vC1,_mm_mul_pd(vA4,vB1));
							vC2 = _mm_add_pd(vC2,_mm_mul_pd(vA4,vB2));
							vC3 = _mm_add_pd(vC3,_mm_mul_pd(vA14,vB1));
							vC4 = _mm_add_pd(vC4,_mm_mul_pd(vA14,vB2));

							_mm_store_pd(&C[iC],vC1);
							_mm_store_pd(&C[iC+2],vC2);
							_mm_store_pd(&C[iiC],vC3);
							_mm_store_pd(&C[iiC+2],vC4);
						}
					}
				}
			}
		}
	}
}


void dgemm_score_main(double *A, double *B, double *&C, double *&D, uint64_t N){
	Time<secs> t;

	//1
	t.start();
	dgemm_base(A,B,C,N);
	double dgemm_base = t.lap("dgemm_base elapsed time in secs");

	//2
	t.start();
	dgemm_reg_reuse(A,B,D,N);
	double dgemm_reg_reuse = t.lap("dgemm_reg_reuse elapsed time in secs");
	cmpResults(A,B,C,D,N,"dgemm_base","dgemm_reg_reuse");
	zeros(D,N);

	//3
	t.start();
	dgemm_reg_blocking(A,B,D,N);
	double dgemm_reg_blocking = t.lap("dgemm_reg_blocking elapsed time in secs");
	cmpResults(A,B,C,D,N,"dgemm_base","dgemm_reg_blocking");
	zeros(D,N);

	//4
	t.start();
	dgemm_reorder(A,B,D,N);
	double dgemm_reorder = t.lap("dgemm_reorder elapsed time in secs");
	cmpResults(A,B,C,D,N,"dgemm_base","dgemm_reorder");
	zeros(D,N);

	t.start();
	dgemm_reorder_sse(A,B,D,N);
	double dgemm_reorder_sse = t.lap("dgemm_reorder_sse elapsed time in secs");
	cmpResults(A,B,C,D,N,"dgemm_base","dgemm_reorder_sse");
	zeros(D,N);

	//5
	t.start();
	dgemm_block(A,B,D,N);
	double dgemm_block = t.lap("dgemm_block elapsed time in secs");
	cmpResults(A,B,C,D,N,"dgemm_base","dgemm_block");
	zeros(D,N);

	//6
	t.start();
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,(int)N,(int)N,(int)N,1.0,A, (int)N, B, (int)N,1.0,D,(int)N);
	double dgemm_cblas = t.lap("dgemm_cblas elapsed time in secs");
	cmpResults(A,B,C,D,N,"dgemm_base","dgemm_cblas");

	double GFLOPS = (double)(N*N*N*2);
	std::cout<< "Elapsed time for dgemm_base: " << dgemm_base << " seconds\n";
	std::cout<< "Elapsed time for dgemm_reg_reuse: " << dgemm_reg_reuse << " seconds\n";
	std::cout<< "Elapsed time for dgemm_reg_blocking: " << dgemm_reg_blocking << " seconds\n";
	std::cout<< "Elapsed time for dgemm_reorder: " << dgemm_reorder << " seconds\n";
	std::cout<< "Elapsed time for dgemm_reorder_sse: " << dgemm_reorder_sse << " seconds\n";
	std::cout<< "Elapsed time for dgemm_block: " << dgemm_block << " seconds\n";
	std::cout<< "Elapsed time for dgemm_cblas: " << dgemm_cblas << " seconds\n";

	std::cout << "GFLOPS for dgemm_base: " << ((double)(GFLOPS/dgemm_base))/1000000000 << "\n";
	std::cout << "GFLOPS for dgemm_reg_reuse: " << ((double)(GFLOPS/dgemm_reg_reuse))/1000000000 << "\n";
	std::cout << "GFLOPS for dgemm_reg_blocking: " << ((double)(GFLOPS/dgemm_reg_blocking))/1000000000 << "\n";
	std::cout << "GFLOPS for dgemm_reorder: " << ((double)(GFLOPS/dgemm_reorder))/1000000000 << "\n";
	std::cout << "GFLOPS for dgemm_reorder_sse: " << ((double)(GFLOPS/dgemm_reorder_sse))/1000000000 << "\n";
	std::cout << "GFLOPS for dgemm_block: " << ((double)(GFLOPS/dgemm_block))/1000000000 << "\n";
	std::cout << "GFLOPS for dgemm_cblas: " << ((double)(GFLOPS/dgemm_cblas))/1000000000 << "\n";
}

#endif
