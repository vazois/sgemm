#ifndef MLAS_VECOP_H
#define MLAS_VECOP_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace mm{
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

	template<typename DATA_T, typename SIZE_T, unsigned int TILE>
	__global__ void sgemm_shared(
			DATA_T *A,
			DATA_T *B,
			DATA_T *C,
			SIZE_T m,
			SIZE_T n,
			SIZE_T k
		){
			__shared__ DATA_T sA[TILE * TILE];
			__shared__ DATA_T sB[TILE * TILE];

			int row = ( blockIdx.y * blockDim.y + threadIdx.y );
			int col = ( blockIdx.x * blockDim.x + threadIdx.x );
			DATA_T rC = 0;

			for(int i = 0; i < (n + 1)/TILE; i++){
					sA[threadIdx.y*TILE + threadIdx.x] = A[ row * (n+1) + i * TILE + threadIdx.x ];
					sB[threadIdx.y*TILE + threadIdx.x] = B[ (i * TILE + threadIdx.y) * k  + col];
					__syncthreads();
					for(int j = 0;j< TILE; j++){
						rC += sA[threadIdx.y * TILE + j] * sB[j * TILE + threadIdx.x]; // 354 GFLOPS
						//rC += sA[threadIdx.y * TILE + j] * sB[threadIdx.y * TILE + j]; // 454 GFLOPS
					}
					__syncthreads();
			}

			C[row * k + col] = rC;
	}
}


#endif
