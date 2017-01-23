#ifndef INIT_CONFIG_H
#define INIT_CONFIG_H

#include <math.h>

double randValue(){
	srand(time(NULL));
	double X=((double)rand()/(double)RAND_MAX);
	//X = roundf(X * 100) / 100;
	X = ceilf(X);

	return X;
}

void init(double *&A, double *&B, unsigned int N){
	for(unsigned int i = 0 ; i < N*N;i++){
		A[i] = randValue();
		B[i] = randValue();
		//A[i] = 1;
		//B[i] = 1;
	}
}

void zeros(double *&C, unsigned int N){
	for(unsigned int i = 0 ; i < N*N;i++){ C[i] = 0; }
}



#endif
