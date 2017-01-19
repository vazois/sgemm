#ifndef INIT_CONFIG_H
#define INIT_CONFIG_H

float randValue(){
	srand(time(NULL));
	float X=((float)rand()/(float)RAND_MAX);

	return X;
}

void init(float *&A, float *&B, unsigned int N){
	for(unsigned int i = 0 ; i < N*N;i++){
		A[i] = randValue();
		B[i] = randValue();
	}
}

void zeros(float *&C, unsigned int N){
	for(unsigned int i = 0 ; i < N*N;i++){ C[i] = 0; }
}

#endif
