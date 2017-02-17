#ifndef SCORE_H
#define SCORE_H

#include <thread>
#include <cmath>
#include <mmintrin.h>
#include <xmmintrin.h>
#include <omp.h>
#include <stdint.h>
#include <cblas.h>
#include "../time/Time.h"

#define THREADS 16

void cmpResults(double *A,double *B, double *C, double *D, uint64_t M, uint64_t K, std::string a, std::string b){
	double diff = std::abs(C[0] - D[0]);
	double maxA = std::abs(C[0]);
	double maxB = std::abs(D[0]);

	double eps = 1.e-6;
	for(uint64_t i = 0; i <M * K; i++){
		if(std::abs(C[i] - D[i]) > diff) diff = std::abs(C[i] - D[i]);
		//if(std::abs(C[i]) > maxA) maxA = std::abs(A[i]);
		//if(std::abs(D[i]) > maxB) maxB = std::abs(B[i]);
	}
	diff/=maxA*maxB;
	if(diff < eps) std::cout << "(PASS) "<< a << " = " << b << std::endl;
	else std::cout << "(ERROR) "<< a << " = " << b << " (" << diff <<")"<< std::endl;
}

void dgemm_base(const double *A, const double *B, double *&C, uint64_t M, uint64_t N, uint64_t K){
	for(uint64_t i = 0; i < M; i++){
		for(uint64_t j = 0; j < K; j++){
			for(uint64_t k = 0; k < N; k++){
				C[i * K + j] += A[ i * N + k ] * B[ k * K + j ];
			}
		}
	}
}

void dgemm_reg_reuse(const double *A, const double *B, double *&C, uint64_t M, uint64_t N, uint64_t K){
	for(uint64_t i = 0; i < M ;i++){
		for(uint64_t j = 0; j < K ;j++){
			register double rC = 0.0f;
			for(uint64_t k = 0; k < N ;k++){
				rC += A[ i * N + k ] * B[ k * K + j ];
			}
			C[i * K + j] = rC;
		}
	}
}

void dgemm_reg_blocking(const double* __restrict__ A, const double* __restrict__ B, double*& __restrict__ C, uint64_t M, uint64_t N, uint64_t K){
	for(uint64_t i = 0; i < M ;i+=4){
		for(uint64_t j = 0; j < K ;j+=4){
			register uint64_t iC = i * N + j;
			register uint64_t iiC = iC + N;
			register uint64_t iiiC = iiC + N;
			register uint64_t iiiiC = iiiC + N;

			register double rC00 = 0.0f;
			register double rC01 = 0.0f;
			register double rC02 = 0.0f;
			register double rC03 = 0.0f;

			register double rC10 = 0.0f;
			register double rC11 = 0.0f;
			register double rC12 = 0.0f;
			register double rC13 = 0.0f;

			register double rC20 = 0.0f;
			register double rC21 = 0.0f;
			register double rC22 = 0.0f;
			register double rC23 = 0.0f;

			register double rC30 = 0.0f;
			register double rC31 = 0.0f;
			register double rC32 = 0.0f;
			register double rC33 = 0.0f;

			for(uint64_t k = 0; k < N ;k+=4){
				register uint64_t iA = i * N + k;
				register uint64_t iiA = iA + N;
				register uint64_t iiiA = iiA + N;
				register uint64_t iiiiA = iiiA + N;

				register uint64_t iB = k * N + j;
				register uint64_t iiB = iB + N;
				register uint64_t iiiB = iiB + N;
				register uint64_t iiiiB = iiiB + N;

				register double rA0 = A[iA]; register double rA1 = A[iiA]; register double rA2 = A[iiiA]; register double rA3 = A[iiiiA];
				register double rB0 = B[iB]; register double rB1 = B[iB + 1]; register double rB2 = B[iB + 2]; register double rB3 = B[iB + 3];
				rC00 += rA0 * rB0; rC10 += rA1 * rB0; rC20 += rA2 * rB0; rC30 += rA3 * rB0;
				rC01 += rA0 * rB1; rC11 += rA1 * rB1; rC21 += rA2 * rB1; rC31 += rA3 * rB1;
				rC02 += rA0 * rB2; rC12 += rA1 * rB2; rC22 += rA2 * rB2; rC32 += rA3 * rB2;
				rC03 += rA0 * rB3; rC13 += rA1 * rB3; rC23 += rA2 * rB3; rC33 += rA3 * rB3;

				rA0 = A[iA+1]; rA1 = A[iiA+1]; rA2 = A[iiiA+1]; rA3 = A[iiiiA+1];
				rB0 = B[iiB]; rB1 = B[iiB + 1]; rB2 = B[iiB + 2]; rB3 = B[iiB + 3];
				rC00 += rA0 * rB0; rC10 += rA1 * rB0; rC20 += rA2 * rB0; rC30 += rA3 * rB0;
				rC01 += rA0 * rB1; rC11 += rA1 * rB1; rC21 += rA2 * rB1; rC31 += rA3 * rB1;
				rC02 += rA0 * rB2; rC12 += rA1 * rB2; rC22 += rA2 * rB2; rC32 += rA3 * rB2;
				rC03 += rA0 * rB3; rC13 += rA1 * rB3; rC23 += rA2 * rB3; rC33 += rA3 * rB3;

				rA0 = A[iA+2]; rA1 = A[iiA+2]; rA2 = A[iiiA+2]; rA3 = A[iiiiA+2];
				rB0 = B[iiiB]; rB1 = B[iiiB + 1]; rB2 = B[iiiB + 2]; rB3 = B[iiiB + 3];
				rC00 += rA0 * rB0; rC10 += rA1 * rB0; rC20 += rA2 * rB0; rC30 += rA3 * rB0;
				rC01 += rA0 * rB1; rC11 += rA1 * rB1; rC21 += rA2 * rB1; rC31 += rA3 * rB1;
				rC02 += rA0 * rB2; rC12 += rA1 * rB2; rC22 += rA2 * rB2; rC32 += rA3 * rB2;
				rC03 += rA0 * rB3; rC13 += rA1 * rB3; rC23 += rA2 * rB3; rC33 += rA3 * rB3;

				rA0 = A[iA+3]; rA1 = A[iiA+3]; rA2 = A[iiiA+3]; rA3 = A[iiiiA+3];
				rB0 = B[iiiiB]; rB1 = B[iiiiB + 1]; rB2 = B[iiiiB + 2]; rB3 = B[iiiiB + 3];
				rC00 += rA0 * rB0; rC10 += rA1 * rB0; rC20 += rA2 * rB0; rC30 += rA3 * rB0;
				rC01 += rA0 * rB1; rC11 += rA1 * rB1; rC21 += rA2 * rB1; rC31 += rA3 * rB1;
				rC02 += rA0 * rB2; rC12 += rA1 * rB2; rC22 += rA2 * rB2; rC32 += rA3 * rB2;
				rC03 += rA0 * rB3; rC13 += rA1 * rB3; rC23 += rA2 * rB3; rC33 += rA3 * rB3;


			}

			C[ iC ] = rC00;
			C[ iC + 1 ] = rC01;
			C[ iC + 2 ] = rC02;
			C[ iC + 3 ] = rC03;

			C[ iiC ] = rC10;
			C[ iiC + 1 ] = rC11;
			C[ iiC + 2 ] = rC12;
			C[ iiC + 3 ] = rC13;

			C[ iiiC ] = rC20;
			C[ iiiC + 1 ] = rC21;
			C[ iiiC + 2 ] = rC22;
			C[ iiiC + 3 ] = rC23;

			C[ iiiiC ] = rC30;
			C[ iiiiC + 1 ] = rC31;
			C[ iiiiC + 2 ] = rC32;
			C[ iiiiC + 3 ] = rC33;
		}
	}
}

void dgemm_reorder(const double* __restrict__ A, const double* __restrict__ B, double*& __restrict__ C, uint64_t M, uint64_t N, uint64_t K){
	for(uint64_t k = 0; k < N ;k+=4){
		for(uint64_t i = 0; i < M ; i++){
			register uint64_t iA = i * N + k;
			register double rA = A[ iA ];
			register double rA1 = A[ iA + 1 ];
			register double rA2 = A[ iA + 2 ];
			register double rA3 = A[ iA + 3 ];

			register double rC = 0.0f;
			for(uint64_t j = 0 ; j < K ; j+=4){
				register uint64_t iB = k * K + j;
				register uint64_t iiB = iB + K;
				register uint64_t iiiB = iiB + K;
				register uint64_t iiiiB = iiiB + K;

				register uint64_t iC = i * K + j;

				C[iC] += rA * B[ iB ] + rA1 * B[ iiB ] + rA2 * B[ iiiB ] + rA3 * B[ iiiiB ];
				C[iC + 1] += rA * B[ iB + 1 ] + rA1 * B[ iiB + 1 ] + rA2 * B[ iiiB + 1 ] + rA3 * B[ iiiiB + 1 ];
				C[iC + 2] += rA * B[ iB + 2 ] + rA1 * B[ iiB + 2 ] + rA2 * B[ iiiB + 2 ] + rA3 * B[ iiiiB + 2 ];
				C[iC + 3] += rA * B[ iB + 3 ] + rA1 * B[ iiB + 3 ] + rA2 * B[ iiiB + 3 ] + rA3 * B[ iiiiB + 3 ];
			}
		}
	}
}

void dgemm_reorder_sse(const double *A, const double *B, double *&C, uint64_t M, uint64_t N, uint64_t K){
	for(uint64_t i = 0; i < M ; i+=2){
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

			for(uint64_t j = 0 ; j < K ; j+=4){
				register uint64_t iB = k * K + j;
				register uint64_t iiB = iB + K;
				register uint64_t iiiB = iiB + K;
				register uint64_t iiiiB = iiiB + K;

				register uint64_t iC = i * K + j;
				register uint64_t iiC = iC + K;


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

void dgemm_block(const double* __restrict__ A, const double* __restrict__  B, double*& __restrict__  C, uint64_t M, uint64_t N, uint64_t K){
	uint64_t Bx = 4;//
	uint64_t By = MIN(1024,K);//
	uint64_t Bz = 4;//
	for(uint64_t j = 0; j < K; j+=By){
		for(uint64_t i = 0; i < M; i+=Bx){
			for(uint64_t k = 0; k < N; k+=Bz){
				//Blocked Matrix

				for(uint64_t kk = k; kk < k + Bz ;kk+=4){
					for(uint64_t ii = i; ii < i + Bx ; ii+=2){
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
							register uint64_t iB = kk * K + jj;
							register uint64_t iiB = iB + K;
							register uint64_t iiiB = iiB + K;
							register uint64_t iiiiB = iiiB + K;

							register uint64_t iC = ii * K + jj;
							register uint64_t iiC = iC + K;

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

void dgemm_block3(const double* __restrict__ A, const double* __restrict__  B, double*& __restrict__  C, uint64_t M, uint64_t N, uint64_t K){
	uint64_t Bx = 4;//
	uint64_t By = MIN(1024,K);//
	uint64_t Bz = 4;//

	for(uint64_t k = 0; k < N; k+=Bz){
		for(uint64_t i = 0; i < M; i+=Bx){
			for(uint64_t j = 0; j < K; j+=By){

				for(uint64_t kk = k; kk < k + Bz; kk++){
					for(uint64_t ii = i; ii < i + Bx; ii++){
						register double rA = A[ii * N + kk];
						for(uint64_t jj = j; jj < j + By; jj++){
							C[ii * K + jj]+= rA * B[kk*K + jj];
						}
					}
				}

			}
		}
	}
}

void dgemm_omp(double *A, double *B, double *&C, uint64_t M, uint64_t N, uint64_t K){
	uint32_t threads = 16;
	uint32_t div = log2(threads);
	omp_set_num_threads(threads);

	#pragma omp parallel
	{
		uint32_t tid = omp_get_thread_num();
		uint32_t gid = omp_get_num_threads();
		uint32_t low = ( tid * M ) / gid;
		uint32_t high = ( (tid+1) * M ) / gid;


		uint64_t Bx = 8;//
		uint64_t By = MIN(1024,K);//
		uint64_t Bz = 8;//
		for(uint64_t k = 0; k < N; k+=Bz){
			for(uint64_t i = low; i < high; i+=Bx){
				for(uint64_t j = 0; j < K; j+=By){
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
								register uint64_t iB = kk * K + jj;
								register uint64_t iiB = iB + K;
								register uint64_t iiiB = iiB + K;
								register uint64_t iiiiB = iiiB + K;

								register uint64_t iC = ii * K + jj;
								register uint64_t iiC = iC + K;

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

	/*#pragma omp parallel
	{
		uint32_t tid = omp_get_thread_num();
		uint32_t gid = omp_get_num_threads();
		uint32_t low = ( tid * M ) / gid;
		uint32_t high = ( (tid+1) * M ) / gid;
		for(uint64_t k = 0; k < N ;k+=4){
			for(uint64_t i = low; i < high ; i+=2){
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

				for(uint64_t j = 0 ; j < K ; j+=4){
					register uint64_t iB = k * K + j;
					register uint64_t iiB = iB + K;
					register uint64_t iiiB = iiB + K;
					register uint64_t iiiiB = iiiB + K;

					register uint64_t iC = i * K + j;
					register uint64_t iiC = iC + K;

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
	}*/
}

void dgemm_func2(const double* __restrict__ A, const double* __restrict__  B, double*& __restrict__  C, uint64_t M, uint64_t N, uint64_t K, uint32_t id, uint32_t group){
	uint64_t Bx = 4;//
	uint64_t By = MIN(1024,K);//
	uint64_t Bz = 4;//

	for(uint64_t k = 0; k < N; k+=Bz){
		for(uint64_t i = 0; i < M; i+=Bx){
			for(uint64_t j = 0; j < K; j+=By){

				for(uint64_t kk = k; kk < k + Bz; kk++){
					for(uint64_t ii = i; ii < i + Bx; ii++){
						register double rA = A[ii * N + kk];
						for(uint64_t jj = j; jj < j + By; jj++){
							C[ii * K + jj]+= rA * B[kk*K + jj];
						}
					}
				}

			}
		}
	}
}

void dgemm_func(double *A, double *B, double *&C, uint64_t M, uint64_t N, uint64_t K, uint32_t id, uint32_t group){
	uint32_t threads = 1;
	uint32_t div = log2(threads);
	omp_set_num_threads(threads);

	uint32_t low = (id*M)/group;
	uint32_t high = ((id+1)*M)/group;

	uint64_t Bx = 8;//
	uint64_t By = MIN(1024,K);//
	uint64_t Bz = 8;//
	for(uint64_t k = 0; k < N; k+=Bz){
		for(uint64_t i = low; i < high; i+=Bx){
			for(uint64_t j = 0; j < K; j+=By){
				//Blocked Matrix

				for(uint64_t kk = k; kk < k + Bz ;kk+=4){
					for(uint64_t ii = i; ii < i + Bx ; ii+=2){
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
							register uint64_t iB = kk * K + jj;
							register uint64_t iiB = iB + K;
							register uint64_t iiiB = iiB + K;
							register uint64_t iiiiB = iiiB + K;

							register uint64_t iC = ii * K + jj;
							register uint64_t iiC = iC + K;

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

void dgemm_pthread(double *A, double *B, double *&C, uint64_t M, uint64_t N, uint64_t K){
	uint32_t threads = 1;
	//std::thread *t = new std::thread(dgemm_omp3,A,B,C,M,N,K,);
	std::thread tarray[THREADS];

	for(uint32_t i = 0; i<THREADS;i++){
		//tarray[i] = std::thread(test,i,THREADS,std::ref(A));
		tarray[i] = std::thread(dgemm_func,A,B,std::ref(C),M,N,K,i,THREADS);
	}

	for(uint32_t i = 0; i < THREADS; i++){
		tarray[i].join();
	}

}

void dgemm_score_main(double *A, double *B, double *&C, double *&D, uint64_t M, uint64_t N,uint64_t K){
	Time<secs> t;
	uint32_t rounds = 10;

	zeros(C,M,K);
	//1
	t.start();
	cblas_dgemm(
			CblasRowMajor,
			CblasNoTrans,
			CblasNoTrans,
			(int)M, (int)K, (int)N, 1.0,
			A, (int)N,
			B, (int)K,
			1.0,
			C,(int)K
	);
	double dgemm_cblas_t = t.lap("dgemm_cblas elapsed time in secs");
	//cmpResults(A,B,C,D,N,"dgemm_base","dgemm_cblas");

	//2
	zeros(D,M,K);
	t.start();
	dgemm_base(A,B,D,M,N,K);
	double dgemm_base_t = t.lap("dgemm_base elapsed time in secs");
	cmpResults(A,B,C,D,M,K,"dgemm_cblas","dgemm_base");
	zeros(D,M,K);

	//3
	t.start();
	dgemm_reg_reuse(A,B,D,M,N,K);
	double dgemm_reg_reuse_t = t.lap("dgemm_reg_reuse elapsed time in secs");
	cmpResults(A,B,C,D,M,K,"dgemm_cblas","dgemm_reg_reuse");
	zeros(D,M,K);

	//4
	t.start();
	dgemm_reg_blocking(A,B,D,M,N,K);
	double dgemm_reg_blocking_t = t.lap("dgemm_reg_blocking elapsed time in secs");
	cmpResults(A,B,C,D,M,K,"dgemm_cblas","dgemm_reg_blocking");
	zeros(D,M,K);
	dgemm_reg_blocking_t = 0;
	for(uint32_t i = 0;i<10;i++){
		t.start();
		dgemm_reg_blocking(A,B,D,M,N,K);
		dgemm_reg_blocking_t += t.lap("dgemm_reg_blocking elapsed time in secs");
	}
	dgemm_reg_blocking_t/=rounds;
	zeros(D,M,K);

	//5
	t.start();
	dgemm_reorder(A,B,D,M,N,K);
	double dgemm_reorder_t = t.lap("dgemm_reorder elapsed time in secs");
	cmpResults(A,B,C,D,M,K,"dgemm_cblas","dgemm_reorder");
	zeros(D,M,K);
	dgemm_reorder_t = 0;
	for(uint32_t i = 0;i<10;i++){
		t.start();
		dgemm_reorder(A,B,D,M,N,K);
		dgemm_reorder_t += t.lap("dgemm_reorder elapsed time in secs");
	}
	dgemm_reorder_t/=rounds;
	zeros(D,M,K);

	//6
	t.start();
	dgemm_reorder_sse(A,B,D,M,N,K);
	double dgemm_reorder_sse_t = t.lap("dgemm_reorder_sse elapsed time in secs");
	cmpResults(A,B,C,D,M,K,"dgemm_cblas","dgemm_reorder_sse");
	zeros(D,M,K);
	dgemm_reorder_sse_t = 0;
	for(uint32_t i = 0;i<10;i++){
		t.start();
		dgemm_reorder_sse(A,B,D,M,N,K);
		dgemm_reorder_sse_t += t.lap("dgemm_reorder_sse elapsed time in secs");
	}
	dgemm_reorder_sse_t/=rounds;
	zeros(D,M,K);

	//7
	t.start();
	dgemm_block(A,B,D,M,N,K);
	double dgemm_block_t = t.lap("dgemm_block elapsed time in secs");
	cmpResults(A,B,C,D,M,K,"dgemm_cblas","dgemm_block");
	zeros(D,M,K);
	dgemm_block_t = 0;
	for(uint32_t i = 0;i<10;i++){
		t.start();
		dgemm_block(A,B,D,M,N,K);
		dgemm_block_t += t.lap("dgemm_block elapsed elapsed time in secs");
	}
	dgemm_block_t/=rounds;
	zeros(D,M,K);

	//8
	t.start();
	dgemm_omp(A,B,D,M,N,K);
	double dgemm_omp_t = t.lap("dgemm_omp elapsed time in secs");
	cmpResults(A,B,C,D,M,K,"dgemm_cblas","dgemm_omp");
	zeros(D,M,K);
	dgemm_omp_t = 0;
	for(uint32_t i = 0;i<10;i++){
		t.start();
		dgemm_omp(A,B,D,M,N,K);
		dgemm_omp_t += t.lap("dgemm_omp elapsed time in secs");
	}
	dgemm_omp_t/=rounds;
	zeros(D,M,K);

	t.start();
	dgemm_pthread(A,B,D,M,N,K);
	double dgemm_pthread_t = t.lap("dgemm_pthread elapsed time in secs");
	cmpResults(A,B,C,D,M,K,"dgemm_cblas","dgemm_pthread");
	zeros(D,M,K);
	dgemm_pthread_t = 0;
	for(uint32_t i = 0;i<10;i++){
		t.start();
		dgemm_pthread(A,B,D,M,N,K);
		dgemm_pthread_t += t.lap("dgemm_pthread elapsed time in secs");
	}
	dgemm_pthread_t/=rounds;
	zeros(D,M,K);

	double GFLOPS = (double)(M*N*K*2);
	std::cout<< "Elapsed time for dgemm_cblas: " << dgemm_cblas_t << " seconds\n";
	std::cout<< "Elapsed time for dgemm_base: " << dgemm_base_t << " seconds\n";
	std::cout<< "Elapsed time for dgemm_reg_reuse: " << dgemm_reg_reuse_t << " seconds\n";
	std::cout<< "Elapsed time for dgemm_reg_blocking: " << dgemm_reg_blocking_t << " seconds\n";
	std::cout<< "Elapsed time for dgemm_reorder: " << dgemm_reorder_t << " seconds\n";
	std::cout<< "Elapsed time for dgemm_reorder_sse: " << dgemm_reorder_sse_t << " seconds\n";
	std::cout<< "Elapsed time for dgemm_block: " << dgemm_block_t << " seconds\n";
	std::cout<< "Elapsed time for dgemm_omp: " << dgemm_omp_t<< " seconds\n";
	std::cout<< "Elapsed time for dgemm_pthread: " << dgemm_pthread_t<< " seconds\n";

	std::cout << "GFLOPS for dgemm_cblas: " << ((double)(GFLOPS/dgemm_cblas_t))/1000000000 << "\n";
	std::cout << "GFLOPS for dgemm_base: " << ((double)(GFLOPS/dgemm_base_t))/1000000000 << "\n";
	std::cout << "GFLOPS for dgemm_reg_reuse: " << ((double)(GFLOPS/dgemm_reg_reuse_t))/1000000000 << "\n";
	std::cout << "GFLOPS for dgemm_reg_blocking: " << ((double)(GFLOPS/dgemm_reg_blocking_t))/1000000000 << "\n";
	std::cout << "GFLOPS for dgemm_reorder: " << ((double)(GFLOPS/dgemm_reorder_t))/1000000000 << "\n";
	std::cout << "GFLOPS for dgemm_reorder_sse: " << ((double)(GFLOPS/dgemm_reorder_sse_t))/1000000000 << "\n";
	std::cout << "GFLOPS for dgemm_block: " << ((double)(GFLOPS/dgemm_block_t))/1000000000 << "\n";
	std::cout << "GFLOPS for dgemm_omp: " << ((double)(GFLOPS/dgemm_omp_t))/1000000000 << "\n";
	std::cout << "GFLOPS for dgemm_pthread: " << ((double)(GFLOPS/dgemm_pthread_t))/1000000000 << "\n";
}

#endif
