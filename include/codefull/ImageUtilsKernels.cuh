/*
 * ImageUtilsKernels.cuh
 *
 *  Created on: Nov 7, 2014
 *      Author: Mehran Maghoumi
 *
 *  Contains kernels and defitions for the ImageUtils class
 */

#ifndef IMAGEUTILSKERNELS_CUH_
#define IMAGEUTILSKERNELS_CUH_

#define HOG_MAGNITUDE_BLOCK_SIZE 16

#define HOG_CELL_SIZE 8
#define HOG_BLOCK_SIZE HOG_CELL_SIZE*2

#define TO_DEGREE(S) (S * 180 / 3.14159265)

#ifndef DIVUP
#define DIVUP(x, y) ((x - 1) / y + 1)
#endif

#define HOG_NUM_BINS 9
#define HOG_NUM_CELLS 4

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "device_types.h"


typedef struct {
		unsigned char center1;
		unsigned char center2;
		char index1;
		char index2;
} BinLocation;

/**
 * Calculates the gradient orientation and magnitude for HoG extraction.
 * For color images, the channel with the maximum gradient magnitude is
 * selected for extraction
 */
__global__ void calculateMagnitudeAndOrientation(const float* srcDiffX, const float* srcDiffY, const int srcPitch,
						   float* dstMagnitude, float* dstOrientation, const int dstPitch,
						   const int width, const int height, const int numChannels) {

	const unsigned int idxX = blockIdx.x * HOG_MAGNITUDE_BLOCK_SIZE + threadIdx.x;
	const unsigned int idxY = blockIdx.y * HOG_MAGNITUDE_BLOCK_SIZE + threadIdx.y;

	if (idxX >= width || idxY >= height)
		return;

	float diffX, diffY;
	float maxHypot = -1;
	float bestDiffX, bestDiffY;

	// Find the maximum magnitude and it's channel index (assuming interleaved pixels)
	for (int i = 0 ; i < numChannels ; i++) {
		diffX = srcDiffX[idxY * srcPitch + numChannels * idxX + i];
		diffY = srcDiffY[idxY * srcPitch + numChannels * idxX + i];
		float hypot = sqrt((float)diffX*diffX + diffY*diffY);

		if (hypot > maxHypot) {
			maxHypot = hypot;
			bestDiffX = diffX;
			bestDiffY = diffY;
		}
	}

	// Assign magnitude
	dstMagnitude[idxY * dstPitch + idxX] = maxHypot;

	// Assign the orientation based on the best channel index
	float angle = bestDiffX == 0 ? 0 : 90 + TO_DEGREE(atan(bestDiffY / (float)bestDiffX));
	dstOrientation[idxY * dstPitch + idxX] = angle;
}

/**
 * Finds the bin location of a given angle for HoG feature extraction
 */
inline __device__ BinLocation findPlace(float &angle) {
	const unsigned char binCenters[9] = {10, 30, 50, 70, 90, 110, 130, 150, 170};
	BinLocation result;

	result.index1 = floor(angle/20 - 0.5);
	result.index2 = ceil(angle/20 - 0.5);

	if (result.index1 == result.index2)
	    result.index1 = result.index1 - 1;

	if (result.index1 < 0)
	    result.index1 = 8;

	if (result.index2 > 8)
	    result.index2 = 0;

	result.center1 = binCenters[result.index1];
	result.center2 = binCenters[result.index2];
	return result;
}

/**
 * Finds the offset of the shared memory for each block when
 * performing HoG feature extraction
 */
inline __device__ unsigned int findSharedOffset(const unsigned int &x, const unsigned int &y) {
	if (x < HOG_CELL_SIZE && y < HOG_CELL_SIZE)
		return 0;
	if (x >= HOG_CELL_SIZE && y < HOG_CELL_SIZE)
		return 18;
	if (x < HOG_CELL_SIZE && y >= HOG_CELL_SIZE)
		return 9;
	if (x >= HOG_CELL_SIZE && y >= HOG_CELL_SIZE)
		return 27;

	printf("findOutOffset: Invalid thread index");
	return 0;
}
/**
 * Extracts the normalized HoG features from the given image. the "features" pointer contains the
 * array of features
 */
__global__ void extractFeatures(const float* magnitude, const float* orientation, const int inputPitch, float* features)
{
	__shared__ float blockHistogram[4 * 9];	// To store the histogram of each cell

	// Collaboratively initialize the value of the shared memory array
	if (threadIdx.y < 4 && threadIdx.x < 9) {
		const unsigned int index = threadIdx.y * 9 + threadIdx.x;
		blockHistogram[index] = 0;
	}

	__syncthreads();

	// Determine the input index of each thread
	const unsigned int idxX = blockIdx.x * HOG_CELL_SIZE + threadIdx.x;
	const unsigned int idxY = blockIdx.y * HOG_CELL_SIZE + threadIdx.y;

	// Determine the address of the histogram for the cell that this thread is processing
	float* hist = &blockHistogram[findSharedOffset(threadIdx.x, threadIdx.y)];
	// Determine the output addresses that this block will write its histograms to
	float* blockFeatures = &features[blockIdx.y * (36* gridDim.x) + blockIdx.x * 36];

	float cMag = magnitude[idxY * inputPitch + idxX];
	float cDir = orientation[idxY * inputPitch + idxX];

	if (cDir <= 10) {
		atomicAdd(hist, (cMag * (10 + cDir) /20));
	}
	else if (cDir >= 170) {
		atomicAdd(&hist[9], (cMag * (190 - cDir) /20));
	}
	else {     // Do bi-linear interpolation
		BinLocation location  = findPlace(cDir);
		atomicAdd(&hist[location.index1], cMag * ((location.center2 - cDir) / 20));
		atomicAdd(&hist[location.index2], cMag * ((cDir - location.center1) / 20));
	}

	__syncthreads(); // Wait for all threads in this block to update their histogram

	// Do normalization and write out the results

	if (threadIdx.y < 4 && threadIdx.x < 9) {
		const unsigned int index = threadIdx.y * 9 + threadIdx.x;
		__shared__ float norm[4*9];

		norm[index] = blockHistogram[index] * blockHistogram[index];

		for (unsigned int stride = 18 ; stride > 1 ; stride>>=1) {
			__syncthreads();
			if (index < stride)
				norm[index] += norm[index + stride];

		}

		if (index == 0) {
			norm[0] += 0.001;
			norm[0] = sqrt(norm[0]);
		}

		__syncthreads();

		float normalized = norm[0];

		blockFeatures[index] = blockHistogram[index] / normalized;
	}
}

// Was supposed to avoid over-computations, but in practice does not really matter
__global__ void extractFeatures2(const float* magnitude, const float* orientation, const int inputPitch, float* features)
{
	__shared__ float blockHistogram[4 * 9];	// To store the histogram of each cell

	// Collaboratively initialize the value of the shared memory array
	if (threadIdx.y < 4 && threadIdx.x < 9) {
		const unsigned int index = threadIdx.y * 9 + threadIdx.x;
		blockHistogram[index] = 0;
	}

	__syncthreads();

	// Determine the input index of each thread
	const unsigned int idxX = blockIdx.x * HOG_BLOCK_SIZE + threadIdx.x;
	const unsigned int idxY = blockIdx.y * HOG_BLOCK_SIZE + threadIdx.y;

	// Determine the address of the histogram for the cell that this thread is processing
	float* hist = &blockHistogram[findSharedOffset(threadIdx.x, threadIdx.y)];
	// Determine the output addresses that this block will write its histograms to
	float* blockFeatures = &features[blockIdx.y * (36* gridDim.x) + blockIdx.x * 36];

	float cMag = magnitude[idxY * inputPitch + idxX];
	float cDir = orientation[idxY * inputPitch + idxX];

	if (cDir <= 10) {
		atomicAdd(hist, (cMag * (10 + cDir) /20));
	}
	else if (cDir >= 170) {
		atomicAdd(&hist[9], (cMag * (190 - cDir) /20));
	}
	else {     // Do bi-linear interpolation
		BinLocation location  = findPlace(cDir);
		atomicAdd(&hist[location.index1], cMag * ((location.center2 - cDir) / 20));
		atomicAdd(&hist[location.index2], cMag * ((cDir - location.center1) / 20));
	}

	__syncthreads(); // Wait for all threads in this block to update their histogram

	// Do normalization and write out the results

	if (threadIdx.y < 4 && threadIdx.x < 9) {
		const unsigned int index = threadIdx.y * 9 + threadIdx.x;
//		__shared__ float norm[4*9];
//
//		norm[index] = blockHistogram[index] * blockHistogram[index];
//
//		for (unsigned int stride = 18 ; stride > 1 ; stride>>=1) {
//			__syncthreads();
//			if (index < stride)
//				norm[index] += norm[index + stride];
//
//		}
//
//		if (index == 0) {
//			norm[0] += 0.001;
//			norm[0] = sqrt(norm[0]);
//		}
//
//		__syncthreads();
//
//		float normalized = norm[0];
//
//		blockFeatures[index] = blockHistogram[index] / normalized;

		blockFeatures[index] = blockHistogram[index];

		// If not the last block, duplicate your cell values into next values as well
		if (index >= 18 && blockIdx.x != gridDim.x - 1) {
			blockFeatures[index + 18] = blockHistogram[index];
		}
	}
}




#endif /* IMAGEUTILSKERNELS_CUH_ */
