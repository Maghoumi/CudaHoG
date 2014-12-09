/*
 * PreciseTimer.cpp
 *
 *  Created on: Nov 7, 2014
 *      Author: mehran
 */

#include "PreciseTimer.h"

namespace codefull {

using namespace std;

void PreciseTimer::start() {
	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);

	cudaEventRecord(startEvent, 0);
}

void PreciseTimer::stopAndLog(std::string message) {
	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);

	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
	cudaEventDestroy(startEvent);
	cudaEventDestroy(stopEvent);
	cerr<< message << " took: " << elapsedTime / 1000 << " seconds"<< endl;
}

} /* namespace codefull */
