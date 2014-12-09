/*
 * Example of how to use the mxGPUArray API in a MEX file.  This example shows
 * how to write a MEX function that takes a gpuArray input and returns a
 * gpuArray output, e.g. B=mexFunction(A).
 *
 * Copyright 2012 The MathWorks, Inc.
 */

/*
 * Example of how to use the mxGPUArray API in a MEX file.  This example shows
 * how to write a MEX function that takes a gpuArray input and returns a
 * gpuArray output, e.g. B=mexFunction(A).
 *
 * Copyright 2012 The MathWorks, Inc.
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
	int dim0 = dims[0];
	int dim1 = dims[1];
	int dim2 = ndim > 2 ? dims[2] : 1;

//	mexPrintf("Passed %d, %d, %d\n", dim0, dim1, dim2);

//	mexPrintf("Calling CUDA...\n");

	std::shared_ptr<float> result = extractHOGFeatures(image, dim0, dim1, dim2);
//	mexPrintf("CUDA called successfully!\n");

	int length = getNumberOfHogFeatures(dim0, dim1);

	plhs[0] = mxCreateNumericMatrix(1, length, mxSINGLE_CLASS, mxREAL);
	float * data = (float *) mxGetData(plhs[0]);

//	mexPrintf("Copying data...\n");

	memcpy((void*) data, (const void*) result.get(), length * sizeof(float));

//	mexPrintf("Done!\n");
}
