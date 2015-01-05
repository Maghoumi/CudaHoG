/*
 * CudaHoG.cu
 *
 *  Created on: Oct 24, 2014
 *      Author: Mehran Maghoumi
 *
 *  HoG feature extraction gateway MEX file for MATLAB
 *
 */
#include "mex.h"
#include <memory>
#include <cstring>

std::shared_ptr<float> extractHOGFeatures(float* inputImage, int width, int height, int numChannels);

#define HOG_NUM_BINS 9
#define HOG_NUM_CELLS 4
#define HOG_CELL_SIZE 8
#define HOG_BLOCK_SIZE HOG_CELL_SIZE*2

inline int getNumberOfHogFeatures(int width, int height) {
	int blockCountX = ((width - HOG_BLOCK_SIZE)/(HOG_CELL_SIZE)) + 1;
	int blockCountY = ((height - HOG_BLOCK_SIZE)/(HOG_CELL_SIZE)) + 1;
	return blockCountX * blockCountY * HOG_NUM_CELLS * HOG_NUM_BINS;
}

/*
 * Host code
 */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{

	if (nrhs != 1)
		mexErrMsgTxt("Input should be only a single matrix containing the image\n");

	if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS)
		mexErrMsgTxt("Input image must be of type SINGLE\n");

	// Parse the dimensionality of the input
	const mwSize *dims = mxGetDimensions(prhs[0]);
	mwSize ndim = mxGetNumberOfDimensions(prhs[0]);

	// Obtain input data and dimensions
	float* image = (float*)mxGetData(prhs[0]);
	int height = dims[0];
	int width = dims[1];
	int numChannels = ndim > 2 ? dims[2] : 1;

	std::shared_ptr<float> result = extractHOGFeatures(image, width, height, numChannels);

	int length = getNumberOfHogFeatures(height, width);

	plhs[0] = mxCreateNumericMatrix(1, length, mxSINGLE_CLASS, mxREAL);
	float * data = (float *) mxGetData(plhs[0]);

	memcpy((void*) data, (const void*) result.get(), length * sizeof(float));
}
