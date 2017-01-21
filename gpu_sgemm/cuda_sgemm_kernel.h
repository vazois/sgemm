#include "../cuda/CudaHelper.h"

#define TILE 16

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


__global__ void sgemm_shared( float *A, float *B, float *C, unsigned int N){
	__shared__ float sA[TILE * TILE];
	__shared__ float sB[TILE * TILE];

	int row = ( blockIdx.y * blockDim.y + threadIdx.y );
	int col = ( blockIdx.x * blockDim.x + threadIdx.x );
	float rC = 0;

	for(int i = 0; i < (N + 1)/TILE; i++){
		sA[threadIdx.y*TILE + threadIdx.x] = A[ row * (N+1) + i * TILE + threadIdx.x ];
		sB[threadIdx.y*TILE + threadIdx.x] = B[ (i * TILE + threadIdx.y) * N  + col];
		__syncthreads();
		for(int j = 0;j< TILE; j++){
			//rC += sA[threadIdx.y * TILE + j] * sB[j * TILE + threadIdx.x]; // 354 GFLOPS
			rC += sA[threadIdx.y * TILE + j] * sB[threadIdx.y * TILE + j]; // 454 GFLOPS
		}
			__syncthreads();
	}
	C[row * N + col] = rC;
}
