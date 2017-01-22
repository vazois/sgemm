#include "../time/CTime.h"
#include "../tools/ArgParser.h"
#include "../tools/InitConfig.h"

#include "score.h"



int main(int argc, char **argv){
	ArgParser ap;
	ap.parseArgs(argc,argv);

	if (ap.count() == 0){
		ap.menu();
	}else{
		if( !ap.exists("-md") ){
			std::cout << "Please provide mode of execution using -md. Run without arguments to get menu options!!!" << std::endl;
			return 1;
		}

		if( !ap.exists("-n") ){
			std::cout << "Please provide matrix dimension using -n. Run without arguments to get menu options!!!" << std::endl;
			return 1;
		}
	}

	unsigned int MD = ap.getInt("-md");
	unsigned int N = ap.getInt("-n");

	float *A = new float[N*N];
	float *B = new float[N*N];
	float *C = new float[N*N];
	float *D = new float[N*N];

	init(A,B,N);
	sgemm_score_main(A,B,C,D,N);


	delete A;
	delete B;
	delete C;
	delete D;



	return 0;
}
