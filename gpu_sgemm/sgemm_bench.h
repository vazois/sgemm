#ifndef CUDA_SGEMM_H
#define CUDA_SGEMM_H

#include "../cuda/CudaHelper.h"
#include "../time/Time.h"

#include "sgemm_kernels.h"
#define TILE 16


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

void sgemm_bench(unsigned int N){
	cutil::setActiveDevice(0);
	float *dA,*dB, *dC;
	uint64_t m = N;
	uint64_t n = N;
	uint64_t k = N;

	cutil::safeMalloc<float,uint64_t>(&dA,sizeof(float) * m*n, "Error allocating device memory for dA");
	cutil::safeMalloc<float,uint64_t>(&dB,sizeof(float) * n*k, "Error allocating device memory for dB");
	cutil::safeMalloc<float,uint64_t>(&dC,sizeof(float) * m*k, "Error allocating device memory for dC");
	cutil::cudaRandInit<float,unsigned int>(dA,N*N);
	cutil::cudaRandInit<float,unsigned int>(dB,N*N);

	dim3 mgrid((m-1)/TILE + 1, (k-1)/TILE + 1, 1);
	dim3 mblock(TILE,TILE,1);
	Time<secs> t;

	//base
	t.start();
	mm::sgemm_base<<<mgrid,mblock>>>(dA,dB,dC,N);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing sgemm");
	double sgemm_base=t.lap("sgemm_base elapsed time in secs");

	//shared
	t.start();
	mm::sgemm_shared<float,unsigned int,TILE><<<mgrid,mblock>>>(dA,dB,dC,m,n,k);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing sgemm");
	double sgemm_shared=t.lap("sgemm_shared elapsed time in secs");

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

	//GFLOPS//
	double GFLOPS = (double)(n*m*k*2);
	std::cout << "GFLOPS for sgemm_base: " << ((double)(GFLOPS/sgemm_base))/1000000000 << "\n";
	std::cout << "GFLOPS for sgemm_shared: " << ((double)(GFLOPS/sgemm_shared))/1000000000 << "\n";
	std::cout << "GFLOPS for sgemm_cublas: " << ((double)(GFLOPS/sgemm_cublas))/1000000000 << "\n";

	cudaFree(dA); cudaFree(dB); cudaFree(dC);
	cudaDeviceReset();
}




#endif
