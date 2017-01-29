#include "../tools/ArgParser.h"

#include "sgemm_bench.h"

int multiplier = 1;

int main(int argc,char **argv){
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

	unsigned int N = ap.getInt("-n");
	sgemm_bench(N);



	return 0;
}
