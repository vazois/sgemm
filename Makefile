CPUC=g++
NVCC=/usr/local/cuda-8.0/bin/nvcc

CEXE=cmain
GEXE=gmain

COPT_FLAGS= -O3 -ffast-math -funroll-loops -mmmx -msse -msse2 -march=amdfam10

NVCC_FLAGS = --ptxas-options=-v -gencode arch=compute_35,code=sm_35 -rdc=true
ARCH = -gencode arch=compute_61,code=sm_61

CUBLAS_LIB = -lcublas_static -lculibos

all: cpu gpu
	
cpu:
	$(CPUC) cpu_sgemm/main.cpp -o $(CEXE) $(COPT_FLAGS)
	
gpu:
	$(NVCC) -std=c++11 $(CUBLAS_LIB) $(ARCH) gpu_sgemm/main.cu -o $(GEXE)
	
clean:
	rm -rf $(CEXE)
	rm -rf $(GEXE)