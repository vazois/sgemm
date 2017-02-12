CPUC=g++
NVCC=/usr/local/cuda-8.0/bin/nvcc

CEXE=cmain
GEXE=gmain

CBLAS_FLAGS = -lgsl -lcblas -lm
COPT_FLAGS= -O3 -ffast-math -funroll-loops -msse -mmmx -msse2
INCLUDE_DIR=-I /home/vzois/git/openblas/ -L/home/vzois/git/openblas/ -lopenblas

NVCC_FLAGS = --ptxas-options=-v -gencode arch=compute_35,code=sm_35 -rdc=true
ARCH = -gencode arch=compute_61,code=sm_61
CUBLAS_LIB = -lcublas_static -lculibos

all: cpu gpu
	
cpu:
	$(CPUC) -std=c++11 cpu_sgemm/main.cpp -o $(CEXE) $(INCLUDE_DIR) $(COPT_FLAGS)
	
gpu:
	$(NVCC) -std=c++11 $(CUBLAS_LIB) $(ARCH) gpu_sgemm/main.cu -o $(GEXE)
	
ptx:
	$(NVCC) -std=c++11 $(CUBLAS_LIB) $(ARCH) -ptx gpu_sgemm/main.cu
	./ptx_chain
	
dryrun:
	$(NVCC) -dryrun -std=c++11 $(CUBLAS_LIB) $(ARCH) gpu_sgemm/main.cu -o $(GEXE) --keep 2>dryrunout
	
clean:
	rm -rf $(CEXE)
	rm -rf $(GEXE)
	rm -rf main.ptx
	rm -rf gmain_*
	rm -rf main.cpp*
	rm -rf *.ii
	rm -rf *.cubin
	rm -rf *.gpu
	rm -rf *.cudafe*
	rm -rf *.o
	rm -rf *fatbin*
	rm -rf *module_id
	