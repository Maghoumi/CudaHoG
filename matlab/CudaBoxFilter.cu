/*
 * CudaBoxFilter.cu
 *
 *  Created on: Oct 24, 2014
 *      Author: Mehran Maghoumi
 *
 *  Box filter gateway MEX file for MATLAB
 *
 */


#include "mex.h"
#include <memory>
#include <cstring>

std::shared_ptr<float> filter (float* inputImage, int dim0, int dim1, int dim2, int filterSize);

/*
 * Host code
 */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
	if (nrhs != 2)
		mexErrMsgTxt("Input should be only a single matrix containing the image and the size of the box filter\n");

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
	int filterSize = (double)mxGetScalar(prhs[1]);

//	mexPrintf("Passed %d, %d, %d\n", dim0, dim1, dim2);

//	mexPrintf("Calling CUDA...\n");

	std::shared_ptr<float> result = filter(image, width, height, numChannels, filterSize);
//	mexPrintf("CUDA called successfully!\n");

	plhs[0] = mxCreateNumericArray(ndim, dims, mxSINGLE_CLASS, mxREAL);
	float * data = (float *) mxGetData(plhs[0]);
//	mexPrintf("Copying data...\n");
	memcpy((void*) data, (const void*) result.get(), width * height * numChannels * sizeof(float));

//	mexPrintf("Done!\n");
}
