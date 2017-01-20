CPUC=g++
GPUC=nvcc

CEXE=cmain
GEXE=gmain

COPT_FLAGS= -O3 -ffast-math -funroll-loops -mmmx -msse -msse2 -march=amdfam10 

all:
	$(CPUC) cpu_sgemm/cpu.cpp -o $(CEXE) $(COPT_FLAGS)
	
cpu:
	$(CPUC) cpu_sgemm/cpu.cpp -o $(CEXE) $(COPT_FLAGS)

clean:
	rm -rf $(CEXE)