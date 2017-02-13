#include "../time/CTime.h"
#include "../tools/ArgParser.h"
#include "../tools/InitConfig.h"
#include "../time/Time.h"
#include "../time/CTime.h"

#include "score.h"
//#include "dgemm_omp.h"


int main(int argc, char **argv){
	ArgParser ap;
	ap.parseArgs(argc,argv);

	if (ap.count() == 0){
		ap.menu();
	}else{
		if( !ap.exists("-n") ){
			std::cout << "Please provide matrix dimension using -n. Run without arguments to get menu options!!!" << std::endl;
			return 1;
		}
	}

	uint64_t N = ap.getInt("-n");

	double *A = new double[N*N];
	double *B = new double[N*N];
	double *C = new double[N*N];
	double *D = new double[N*N];

	openblas_set_num_threads(1);
	init(A,B,N);
	dgemm_score_main(A,B,C,D,N);

	//dgemm_omp(A,B,C,N);

	delete A;
	delete B;
	delete C;
	delete D;



	return 0;
}
