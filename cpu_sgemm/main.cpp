#include "../time/CTime.h"
#include "../tools/ArgParser.h"
#include "../tools/InitConfig.h"
#include "../time/Time.h"
#include "../time/CTime.h"

#include <inttypes.h>
#include "score.h"



int main(int argc, char **argv){
	ArgParser ap;
	ap.parseArgs(argc,argv);

	/*if (ap.count() == 0){
		ap.menu();
	}else{
		if( !ap.exists("-n") ){
			std::cout << "Please provide matrix dimension using -n. Run without arguments to get menu options!!!" << std::endl;
			return 1;
		}
	}*/

	//uint64_t M = 128;
	//uint64_t N = 128;
	//uint64_t K = 1024;
	uint64_t M = 1024;
	uint64_t N = 1024;
	uint64_t K = 1024;

	//printf("%" PRIu64 "\n", M);
	printf("C(%" PRId64 ",%" PRId64 ") = A (%" PRId64 ",%" PRId64 ") x B(%" PRId64 ",%" PRId64 ")\n",M,K,M,N,N,K);
	double *A = new double[M*N];
	double *B = new double[N*K];
	double *C = new double[M*K];
	double *D = new double[M*K];

	openblas_set_num_threads(16);
	init(A,B,M,N,K);
	dgemm_score_main(A,B,C,D,M,N,K);

	delete A;
	delete B;
	delete C;
	delete D;



	return 0;
}
