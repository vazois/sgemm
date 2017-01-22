#first make

CC = g++
NVCC = /usr/local/cuda-8.0/bin/nvcc
EXEC = mlas

#LIBRARIES#
LIB_FLAGS = -lcuda -lcudart
INCLUDE_PATHS = /usr/local/cuda/include
INCUDE_LIB = /usr/local/cuda-8.0/lib64
CUBLAS_LIB = -lcublas_static -lculibos

#COMPILATION FILES#
CPP_FILES = *.cpp
CU_FILES = *.cu

NVCC_FLAGS = --ptxas-options=-v -gencode arch=compute_35,code=sm_35
ARCH= -gencode arch=compute_61,code=sm_61

all: cc

cc:
	$(NVCC) -std=c++11 $(CUBLAS_LIB) $(ARCH) *.cu -o $(EXEC)
		
ptx:
	$(NVCC) -std=c++11 $(CUBLAS_LIB) $(ARCH) -ptx *.cu
	
clean:
	rm -rf *.o
	rm -rf $(EXEC)
	rm -rf *.ptx
