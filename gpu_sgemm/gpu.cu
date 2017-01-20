#include "../time/CTime.h"
#include "../tools/ArgParser.h"
#include "../tools/InitConfig.h"
#include "../cuda/CudaHelper.h"

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

	return 0;
}
