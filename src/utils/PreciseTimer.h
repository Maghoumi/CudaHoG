/*
 * PreciseTimer.h
 *
 *  Created on: Nov 7, 2014
 *      Author: Mehran Maghoumi
 *
 *  Provides easy access to CUDA timing functions.
 *  Use objects of this class to measure kernel run times.
 */

#ifndef PRECISETIMER_H_
#define PRECISETIMER_H_

#include <string>
#include <iostream>

#include "commons.h"

namespace codefull {

class PreciseTimer {
protected:
	cudaEvent_t startEvent;
	cudaEvent_t stopEvent;

public:
	PreciseTimer(){}
	virtual ~PreciseTimer(){}

	void start();
	void stopAndLog(std::string message);

};

} /* namespace codefull */
#endif /* PRECISETIMER_H_ */
