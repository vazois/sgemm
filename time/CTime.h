#ifndef CTIME_H
#define CTIME_H

#include <string>
#include <time.h>
#include <iostream>

clock_t start;
clock_t diff;

void start_clock(){
	start = clock();
}

void stop_clock(){
	diff = clock() - start;
}

double sec(std::string msg){
	double tt = diff;
	tt/=CLOCKS_PER_SEC;
	std::cout << msg << " " << tt <<std::endl;
	return tt;
}

double secf(){
	double tt = diff;
	tt/=CLOCKS_PER_SEC;
	return tt;
}

void printTime(double tt, std::string msg){
	std::cout<< msg << " " << tt << std::endl;
}

#endif
