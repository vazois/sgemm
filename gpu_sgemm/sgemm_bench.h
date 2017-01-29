#ifndef CUDA_SGEMM_H
#define CUDA_SGEMM_H

#include "../cuda/CudaHelper.h"
#include "../time/Time.h"
#include "../tools/InitConfig.h"

#include "sgemm_kernels.h"
#define TILE 32
#define TW 4

void cublas_bench(){
	cutil::setActiveDevice(0);
	uint64_t m,n,k;
	m = 1024; n = 1024; k = 1024;

	float *dA,*dB,*dC;
	cutil::safeMalloc<float,uint64_t>(&dA,sizeof(float)*m*n,"dA memory alloc");
	cutil::safeMalloc<float,uint64_t>(&dB,sizeof(float)*n*k,"dB memory alloc");
	cutil::safeMalloc<float,uint64_t>(&dC,sizeof(float)*m*k,"dC memory alloc");

	cutil::cudaRandInit<float,uint64_t>(dA,m*n);
	cutil::cudaRandInit<float,uint64_t>(dB,n*k);

	cublasHandle_t handle;
	cutil::cublasCheckErr(cublasCreate(&handle), "Creating Handle Error!");
	const float alpha = 1.0f;
	const float beta  = 0.0f;

	Time<secs> t;
	t.start();
	cublasSgemm( handle, CUBLAS_OP_N,CUBLAS_OP_N, k,m,n, &alpha, dB,k, dA,n, &beta, dC,k );
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing cublas sgemm");
	double tt=t.lap("sgemm elapsed time in ms");

	uint64_t flop = 2 * ((uint64_t)m)*((uint64_t)n)*((uint64_t)m);
	std::cout << "FLOP:" << flop << std::endl;
	double gflops = ((flop)/(tt))/1000000000;
	std::cout << "GFLOPS CUBLAS sgemm:" << gflops << std::endl;

	cutil::cublasCheckErr(cublasDestroy(handle), "Destroying Handle Error");
	cudaDeviceReset();
}

void sgemm_cache(const float *A, const float *B, float *&C, uint64_t N){
	for(uint64_t i = 0; i < N ; i++){
		for(uint64_t k = 0; k < N ;k+=4){
			register uint64_t iA = i * N + k;
			register float rA = A[ iA ];
			register float rA1 = A[ iA + 1 ];
			register float rA2 = A[ iA + 2 ];
			register float rA3 = A[ iA + 3 ];

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

void cmpResults(float *A,float *B, float *C, float *D, uint64_t N, std::string a, std::string b){
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

void sgemm_bench(unsigned int N){
	cutil::setActiveDevice(0);
	float *dA,*dB, *dC, *dD;
	float *hA,*hB, *hC, *hD;
	uint64_t m = N;
	uint64_t n = N;
	uint64_t k = N;
	Time<secs> t;

	//Alocate memory space//
	cutil::safeMalloc<float,uint64_t>(&dA,sizeof(float) * m*n, "Error allocating device memory for dA");
	cutil::safeMalloc<float,uint64_t>(&dB,sizeof(float) * n*k, "Error allocating device memory for dB");
	cutil::safeMalloc<float,uint64_t>(&dC,sizeof(float) * m*k, "Error allocating device memory for dC");
	cutil::safeMalloc<float,uint64_t>(&dD,sizeof(float) * m*k, "Error allocating device memory for dD");

	cutil::safeMallocHost<float,uint64_t>(&hA,sizeof(float)*m*n, "Error allocating host memory for hA");
	cutil::safeMallocHost<float,uint64_t>(&hB,sizeof(float)*n*k, "Error allocating host memory for hB");
	cutil::safeMallocHost<float,uint64_t>(&hC,sizeof(float)*n*k, "Error allocating host memory for hC");
	cutil::safeMallocHost<float,uint64_t>(&hD,sizeof(float)*n*k, "Error allocating host memory for hD");

	//CPU Comparison
	initF(hA,hB,N);
	cutil::safeCopyToDevice<float,uint64_t>(dA,hA,sizeof(float)*m*n, "Error copying from hA to dA");
	cutil::safeCopyToDevice<float,uint64_t>(dB,hB,sizeof(float)*m*n, "Error copying from hB to dB");
	sgemm_cache(hA,hB,hC,N);


	///////////////////////////////////////////////////////////////
	//base
	dim3 dimBlock(TILE, TILE);
	dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);
	t.start();
	mm::sgemm_base<<<dimGrid,dimBlock>>>(dA,dB,dC,N);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing sgemm");
	double sgemm_base=t.lap("sgemm_base elapsed time in secs");
	cutil::safeCopyToHost(hD, dC, sizeof(float)*N*N, "Error copying dC to hD");
	cmpResults(hA,hB,hC,hD,N,"sgemm_base","sgemm_cache");

	///////////////////////////////////////////////////////////////
	//shared
	dim3 mgrid((k-1)/TILE + 1, (m-1)/TILE + 1, 1);
	dim3 mblock(TILE,TILE,1);
	t.start();
	mm::sgemm_shared<float,unsigned int,TILE><<<mgrid,mblock>>>(dA,dB,dC,m,n,k);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing sgemm");
	double sgemm_shared=t.lap("sgemm_shared elapsed time in secs");
	cutil::safeCopyToHost(hD, dC, sizeof(float)*N*N, "Error copying dC to hD");
	cmpResults(hA,hB,hC,hD,N,"sgemm_shared","sgemm_cache");


	///////////////////////////////////////////////////////////////
	//shared2
	dim3 kgrid((k-1)/(TILE*TW) + 1, (m-1)/(TILE) + 1, 1);
	dim3 kblock(TILE,TILE,1);
	cutil::print_grid(kgrid,kblock,"kgpu");
	t.start();
	mm::sgemm_shared2<float,unsigned int,TILE,TW><<<kgrid,kblock>>>(dA,dB,dC,m,n,k);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing sgemm");
	double sgemm_shared2=t.lap("sgemm_shared2 elapsed time in secs");
	cutil::safeCopyToHost(hD, dC, sizeof(float)*N*N, "Error copying dC to hD");
	cmpResults(hA,hB,hC,hD,N,"sgemm_shared2","sgemm_cache");


	///////////////////////////////////////////////////////////////
	//cublas
	cublasHandle_t handle;
	cutil::cublasCheckErr(cublasCreate(&handle), "Creating Handle Error!");
	const float alpha = 1.0f;
	const float beta  = 0.0f;
	t.start();
	cublasSgemm( handle, CUBLAS_OP_N,CUBLAS_OP_N, k,m,n, &alpha, dB,k, dA,n, &beta, dC,k );
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing cublas sgemm");
	double sgemm_cublas=t.lap("sgemm_cublas elapsed time in secs");
	cutil::cublasCheckErr(cublasDestroy(handle), "Destroying Handle Error");


	///////////////////////////////////////////////////////////////
	//GFLOPS//
	double GFLOPS = (double)(n*m*k*2);
	std::cout << "GFLOPS for sgemm_base: " << ((double)(GFLOPS/sgemm_base))/1000000000 << "\n";
	std::cout << "GFLOPS for sgemm_shared: " << ((double)(GFLOPS/sgemm_shared))/1000000000 << "\n";
	std::cout << "GFLOPS for sgemm_shared2: " << ((double)(GFLOPS/sgemm_shared2))/1000000000 << "\n";
	std::cout << "GFLOPS for sgemm_cublas: " << ((double)(GFLOPS/sgemm_cublas))/1000000000 << "\n";
	///////////////////////////////////////////////////////////////

	cudaFree(dA); cudaFree(dB); cudaFree(dC);
	cudaFreeHost(hA); cudaFreeHost(hB); cudaFreeHost(hC); cudaFreeHost(hD);
	cudaDeviceReset();
}




#endif
