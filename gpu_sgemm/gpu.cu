#include "../time/CTime.h"
#include "../tools/ArgParser.h"
#include "../tools/InitConfig.h"
#include "../cuda/CudaHelper.h"

#include "cuda_sgemm_kernel.h"


int main(int argc, char **argv){
	ArgParser ap;
	ap.parseArgs(argc,argv);

	if (ap.count() == 0){
		ap.menu();
	}else{
		if( !ap.exists("-md") ){
			std::cout << "Please provide mode of execution using -md. Run without arguments to get menu options!!!" << std::endl;
			return 1;
		}

		if( !ap.exists("-n") ){
			std::cout << "Please provide matrix dimension using -n. Run without arguments to get menu options!!!" << std::endl;
			return 1;
		}
	}

	printf("Starting Execution!!!\n");
	unsigned int MD = ap.getInt("-md");
	unsigned int N = ap.getInt("-n");
	float *devA,*devB, *devC, *devD;

	cutil::safeMallocHost<float,unsigned int>(&devA,sizeof(float)*N*N,"error allocating devA memory space");
	cutil::safeMallocHost<float,unsigned int>(&devB,sizeof(float)*N*N,"error allocating devA memory space");
	cutil::safeMallocHost<float,unsigned int>(&devC,sizeof(float)*N*N,"error allocating devA memory space");
	cutil::safeMallocHost<float,unsigned int>(&devD,sizeof(float)*N*N,"error allocating devA memory space");

	cutil::cudaRandInit<float,unsigned int>(devA,N*N);
	cutil::cudaRandInit<float,unsigned int>(devB,N*N);

	dim3 mgrid((N-1)/TILE + 1, (N-1)/TILE + 1, 1);
	dim3 mblock(TILE,TILE,1);

	sgemm_base<<<mgrid,mblock>>>(devA,devB,devC,N);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing sgemm_base");

	sgemm_shared<<<mgrid,mblock>>>(devA,devB,devC,N);
	cutil::cudaCheckErr(cudaDeviceSynchronize(),"Error executing sgemm_shared");

	cudaFree(devA); cudaFree(devB); cudaFree(devC); cudaFree(devD);
	cudaDeviceReset();

	return 0;
}
