# Matrix Multiplication Performance Optimization
- CPU Optimized Implementation
- GPU Optimized Implementation
Although it is called sgemm the CPU uses double precision


#Compile Instruction
Type make cpu to compile the cpu GEMM
Type make gpu to compile the gpu GEMM
Type make to compile both implementation
Use source ./setup.sh to add cblas to the LD_LIBRARY_PATH

#Execution instruction
Use dimension larger than 128. Every matrix is assumed to be square for simplicity.

#CPU Execute Benchmark
./cmain -n=1024 to multiply two random 1024x1024 matrices using the CPU (16 threads)
Change the number of threads in cpu_sgemm score.h

#GPU Execute Benchmark
./gmain -n=1024 to multiply two random 1024x1024 matrices using the GPU

