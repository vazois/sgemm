
#include <cmath>

//#include <omp.h>

#define USE_OPENMP

void dgemm_omp(double *A, double *B, double *&C, uint64_t N){
	uint32_t threads = 16;
	uint32_t div = log2(threads);
	omp_set_num_threads(threads);

	#pragma omp parallel
	{
		uint32_t tid = omp_get_thread_num();
		uint32_t row = tid / (omp_get_num_threads() / div);
		uint32_t col = tid & ((omp_get_num_threads() / div) - 1);

		printf("%d,%d,%d\n",tid,row,col);
	}

}
