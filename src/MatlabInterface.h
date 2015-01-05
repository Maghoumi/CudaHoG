/*
 * MatlabInterface.h
 *
 *  Created on: Dec 09, 2014
 *      Author: Mehran Maghoumi
 *
 *  Defines the interface for MEX.  MATLAB USE ONLY!
 *	Defines the entry points for the MATLAB interface
 */

#ifndef MATLABINTERFACE_H_
#define MATLABINTERFACE_H_

#include "codefull/ImageUtils.h"
#include "codefull/CudaMemory.h"
#include "codefull/FloatImage.h"

using namespace codefull;


std::shared_ptr<float> extractHOGFeatures(float* inputImage, int dim0, int dim1, int dim2);

std::shared_ptr<float> filter (float* inputImage, int dim0, int dim1, int dim2, int filterSize);

#endif /* MATLABINTERFACE_H_ */
