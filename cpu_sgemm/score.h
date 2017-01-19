#ifndef SCORE_H
#define SCORE_H

void sgemm_base(const float *A, const float *B, float *&C, unsigned int N){
	for(unsigned int i = 0; i < N ;i++){
		for(unsigned int j = 0; j < N ;j++){
			for(unsigned int k = 0; k < N ;k++){
				C[i * N + j] += A[ i * N + k ] * B[ k * N + j ];
			}
		}
	}
}


void sgemm_reg_acc(const float *A, const float *B, float *&C, unsigned int N){
	for(unsigned int i = 0; i < N ;i++){
		for(unsigned int j = 0; j < N ;j++){
			float rC = 0.0f;
			for(unsigned int k = 0; k < N ;k++){
				rC += A[ i * N + k ] * B[ k * N + j ];
			}
			C[i * N + j] = rC;
		}
	}
}


void sgemm_score(const float *A, const float *B, float *&C, float *&D, unsigned int N){

	//1
	start_clock();
	sgemm_base(A,B,C,N);
	stop_clock();
	double sgemm_base = secf();

	//2
	start_clock();
	sgemm_reg_acc(A,B,D,N);
	stop_clock();
	double sgemm_reg_acc = secf();

	std::cout<< "Elapsed time for sgemm_base: " << sgemm_base << " seconds\n";
	std::cout<< "Elapsed time for sgemm_reg_acc: " << sgemm_reg_acc << " seconds\n";
}

#endif
