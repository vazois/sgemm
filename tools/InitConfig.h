#ifndef INIT_CONFIG_H
#define INIT_CONFIG_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double randValue(){
	double X=((double)rand()*128/(double)RAND_MAX) + 1.0f;
	return roundf(X);
}

void init(double *&A, double *&B, unsigned int N){
	srand(time(NULL));
	for(unsigned int i = 0 ; i < N*N;i++){
		A[i] = randValue();
		B[i] = randValue();
		//A[i] = 1;
		//B[i] = 1;
	}
}

float randValueF(){
	float X=(float)rand()*4/(float)(RAND_MAX) + 1.0f;
	return roundf(X);
}

void initF(float *&A, float *&B, uint64_t N){
	srand(time(NULL));
	for(uint64_t i = 0 ; i < N*N;i++){
		A[i] = randValueF();
		B[i] = randValueF();
	}
}

void zeros(double *&C, unsigned int N){
	for(unsigned int i = 0 ; i < N*N;i++){ C[i] = 0; }
}



#endif
