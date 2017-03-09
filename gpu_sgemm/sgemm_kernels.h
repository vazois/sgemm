#ifndef MLAS_VECOP_H
#define MLAS_VECOP_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace mm{

	__global__ void zeros(float *C, uint64_t N){
		uint64_t row = blockDim.y * blockIdx.y + threadIdx.y;
		uint64_t col = blockDim.x * blockIdx.x + threadIdx.x;

		if ( row < N && col < N){
			C[ row * N + col ] = 0.0f;
		}
	}

	__global__ void sgemm_base(float *A, float *B, float *C, unsigned int N){
		float rC = 0.0f;
		unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
		unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

		if ( row < N && col < N){
			for (unsigned int i = 0;i < N ; i++){
				rC += A[row * N + i] * B[i * N + col];
			}

			C[ row * N + col ] = rC;
		}
	}

	__global__ void sgemm_base2(float *A, float *B, float *C, unsigned int N){
			float rC = 0.0f;
			unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
			unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

			if ( row < N && col < N){
			for (unsigned int i = 0;i < N ; i++){
				rC += A[row * N + i] * B[i * N + col];
			}
			C[ row * N + col ] = rC;
		}
	}


	template<typename DATA_T, typename SIZE_T, unsigned int TILE>
	__global__ void sgemm_shared(
			DATA_T *A,
			DATA_T *B,
			DATA_T *C,
			SIZE_T m,
			SIZE_T n,
			SIZE_T k
		){
			__shared__ DATA_T sA[TILE][TILE];
			__shared__ DATA_T sB[TILE][TILE];

			int row = ( blockIdx.y * blockDim.y + threadIdx.y );
			int col = ( blockIdx.x * blockDim.x + threadIdx.x );
			DATA_T rC = 0;

			for(int i = 0; i < (n-1)/TILE + 1; i++){
					sA[threadIdx.y][threadIdx.x] = A[ row * n + i * TILE + threadIdx.x ];
					sB[threadIdx.y][threadIdx.x] = B[ (i * TILE + threadIdx.y) * k  + col];
					__syncthreads();
					for(int j = 0;j< TILE; j++){
						rC += sA[threadIdx.y][j] * sB[j][threadIdx.x]; // 354 GFLOPS
					}
					__syncthreads();
			}

			C[row * k + col] = rC;
	}

	template<typename DATA_T, typename SIZE_T, unsigned int TILE, unsigned int TW>
	__global__ void sgemm_shared2(
			DATA_T *A,
			DATA_T *B,
			DATA_T *C,
			SIZE_T m,
			SIZE_T n,
			SIZE_T k
		){
			__shared__ DATA_T sA[TILE * TILE];
			__shared__ DATA_T sB1[TILE * TILE];
			__shared__ DATA_T sB2[TILE * TILE];
			__shared__ DATA_T sB3[TILE * TILE];
			__shared__ DATA_T sB4[TILE * TILE];
			__shared__ DATA_T sB5[TILE * TILE];
			__shared__ DATA_T sB6[TILE * TILE];
			__shared__ DATA_T sB7[TILE * TILE];
			__shared__ DATA_T sB8[TILE * TILE];

			uint32_t row = ( blockIdx.y * blockDim.y + threadIdx.y );
			uint32_t col = ( blockIdx.x * blockDim.x*TW + threadIdx.x );
			DATA_T rC1 = 0;
			DATA_T rC2 = 0;
			DATA_T rC3 = 0;
			DATA_T rC4 = 0;
			DATA_T rC5 = 0;
			DATA_T rC6 = 0;
			DATA_T rC7 = 0;
			DATA_T rC8 = 0;

			uint32_t sharedLoadIndex = threadIdx.y*TILE + threadIdx.x;
			uint32_t rowOffset = row * n + threadIdx.x;
			uint32_t colOffset = threadIdx.y * k + col;
			for(uint32_t i = 0; i < (n-1)/TILE + 1; i++){
				sA[sharedLoadIndex] = A[ rowOffset ];
				sB1[sharedLoadIndex] = B[ colOffset ];
				sB2[sharedLoadIndex] = B[ colOffset + blockDim.x];
				sB3[sharedLoadIndex] = B[ colOffset + blockDim.x*2];
				sB4[sharedLoadIndex] = B[ colOffset + blockDim.x*3];
				sB5[sharedLoadIndex] = B[ colOffset + blockDim.x*4];
				sB6[sharedLoadIndex] = B[ colOffset + blockDim.x*5];
				sB7[sharedLoadIndex] = B[ colOffset + blockDim.x*6];
				sB8[sharedLoadIndex] = B[ colOffset + blockDim.x*7];
				rowOffset+=TILE;
				colOffset+=k*TILE;
				__syncthreads();
				uint32_t sharedIndexA = threadIdx.y * TILE;
				uint32_t sharedIndexB = threadIdx.x;
				for(uint32_t j = 0;j< TILE; j++){
					rC1 += sA[sharedIndexA] * sB1[sharedIndexB]; // 354 GFLOPS
					rC2 += sA[sharedIndexA] * sB2[sharedIndexB]; // 354 GFLOPS
					rC3 += sA[sharedIndexA] * sB3[sharedIndexB]; // 354 GFLOPS
					rC4 += sA[sharedIndexA] * sB4[sharedIndexB]; // 354 GFLOPS
					rC5 += sA[sharedIndexA] * sB5[sharedIndexB];
					rC6 += sA[sharedIndexA] * sB6[sharedIndexB];
					rC7 += sA[sharedIndexA] * sB7[sharedIndexB];
					rC8 += sA[sharedIndexA] * sB8[sharedIndexB];
					sharedIndexB+=TILE;
					sharedIndexA++;
				}
				__syncthreads();
			}

			C[row * k + col] = rC1;
			C[row * k + col + blockDim.x] = rC2;
			C[row * k + col + blockDim.x*2] = rC3;
			C[row * k + col + blockDim.x*3] = rC4;
			C[row * k + col + blockDim.x*4] = rC5;
			C[row * k + col + blockDim.x*5] = rC6;
			C[row * k + col + blockDim.x*6] = rC7;
			C[row * k + col + blockDim.x*7] = rC8;
	}

	template<typename DATA_T, typename SIZE_T, unsigned int TILE, unsigned int TW>
		__global__ void sgemm_shared3(
				DATA_T *A,
				DATA_T *B,
				DATA_T *C,
				SIZE_T m,
				SIZE_T n,
				SIZE_T k
			){
			__shared__ DATA_T sA[TILE * TILE];
			__shared__ DATA_T sB1[TILE * TILE];
			__shared__ DATA_T sB2[TILE * TILE];
			__shared__ DATA_T sB3[TILE * TILE];
			__shared__ DATA_T sB4[TILE * TILE];
			__shared__ DATA_T sB5[TILE * TILE];
			__shared__ DATA_T sB6[TILE * TILE];
			__shared__ DATA_T sB7[TILE * TILE];
			__shared__ DATA_T sB8[TILE * TILE];

			uint32_t row = ( blockIdx.y * blockDim.y + threadIdx.y );
			uint32_t col = ( blockIdx.x * blockDim.x*TW + threadIdx.x );
			DATA_T rC1 = 0;
			DATA_T rC2 = 0;
			DATA_T rC3 = 0;
			DATA_T rC4 = 0;
			DATA_T rC5 = 0;
			DATA_T rC6 = 0;
			DATA_T rC7 = 0;
			DATA_T rC8 = 0;

			uint32_t sharedLoadIndex = threadIdx.y*TILE + threadIdx.x;
			uint32_t rowOffset = row * n + threadIdx.x;
			uint32_t colOffset = threadIdx.y * k + col;
			for(uint32_t i = 0; i < (n-1)/TILE + 1; i++){
				sA[sharedLoadIndex] = A[ rowOffset ];
				sB1[sharedLoadIndex] = B[ colOffset ];
				sB2[sharedLoadIndex] = B[ colOffset + blockDim.x];
				sB3[sharedLoadIndex] = B[ colOffset + blockDim.x*2];
				sB4[sharedLoadIndex] = B[ colOffset + blockDim.x*3];
				sB5[sharedLoadIndex] = B[ colOffset + blockDim.x*4];
				sB6[sharedLoadIndex] = B[ colOffset + blockDim.x*5];
				sB7[sharedLoadIndex] = B[ colOffset + blockDim.x*6];
				sB8[sharedLoadIndex] = B[ colOffset + blockDim.x*7];
				rowOffset+=TILE;
				colOffset+=k*TILE;
				__syncthreads();
				uint32_t sharedIndexA = threadIdx.y * TILE;
				uint32_t sharedIndexB = threadIdx.x;

				for(uint32_t j = 0;j< TILE; j++){
					rC1 += sA[sharedIndexA] * sB1[sharedIndexB]; // 354 GFLOPS
					rC2 += sA[sharedIndexA] * sB2[sharedIndexB]; // 354 GFLOPS
					rC3 += sA[sharedIndexA] * sB3[sharedIndexB]; // 354 GFLOPS
					rC4 += sA[sharedIndexA] * sB4[sharedIndexB]; // 354 GFLOPS
					rC5 += sA[sharedIndexA] * sB5[sharedIndexB];
					rC6 += sA[sharedIndexA] * sB6[sharedIndexB];
					rC7 += sA[sharedIndexA] * sB7[sharedIndexB];
					rC8 += sA[sharedIndexA] * sB8[sharedIndexB];
					sharedIndexB+=TILE;
					sharedIndexA++;
				}
				__syncthreads();
			}

			C[row * k + col] = rC1;
			C[row * k + col + blockDim.x] = rC2;
			C[row * k + col + blockDim.x*2] = rC3;
			C[row * k + col + blockDim.x*3] = rC4;
			C[row * k + col + blockDim.x*4] = rC5;
			C[row * k + col + blockDim.x*5] = rC6;
			C[row * k + col + blockDim.x*6] = rC7;
			C[row * k + col + blockDim.x*7] = rC8;
	}
}


#endif
