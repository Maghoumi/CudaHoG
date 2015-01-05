/*
 * ImageUtils.cu
 *
 *  Created on: Oct 24, 2014
 *      Author: Mehran Maghoumi
 *
 */

#include "codefull/ImageUtils.h"
#include "codefull/ImageUtilsKernels.cuh"

using namespace std;
using namespace codefull;

std::vector<Npp32s> codefull::ImageUtils::getCentralDifferenceKernel() {
	std::vector<Npp32s> result = {1, 0, -1};
	return result;
}

std::vector<Npp32f> codefull::ImageUtils::getCentralDifferenceKernelFloat() {
	std::vector<Npp32f> result = {1, 0, -1};
	return result;
}

CudaMemory<Npp32f> codefull::ImageUtils::getBoxKernel(const int size) {
	CudaMemory<Npp32f> result(size, size, 1, false);
	float size2 = size * size;
	Npp32f* data = result.getMutableHostData();

	for (int i = 0 ; i < size * size ; i++) {
		data[i] = 1/size2;
	}

	return result;
}

codefull::CudaMemory<float> codefull::ImageUtils::extractHOGFeatures(codefull::FloatImage &image) {
	// Step 1: Find the image gradients
	FloatImage gradX = image.convolve(ImageUtils::getCentralDifferenceKernelX());
	FloatImage gradY = image.convolve(ImageUtils::getCentralDifferenceKernelY());

	CudaMemory<float> magnitude(image.getWidth(), image.getHeight(), 1);magnitude.getMutableDeviceData();
	CudaMemory<float> orientation(image.getWidth(), image.getHeight(), 1);orientation.getMutableDeviceData();

	dim3 gridDim(DIVUP(image.getWidth(), HOG_MAGNITUDE_BLOCK_SIZE), DIVUP(image.getHeight(), HOG_MAGNITUDE_BLOCK_SIZE), 1);
	dim3 blockDim(HOG_MAGNITUDE_BLOCK_SIZE, HOG_MAGNITUDE_BLOCK_SIZE, 1);


	CHECK_CUDA_KERNEL((
	calculateMagnitudeAndOrientation<<<gridDim, blockDim>>>(gradX.getDeviceData(), gradY.getDeviceData(), gradX.getDevicePitchInElements(),
			 magnitude.getMutableDeviceData(), orientation.getMutableDeviceData(), magnitude.getDevicePitchInElements(),
			 image.getWidth(), image.getHeight(), image.getNumChannels())
			 ));

	// Step 2: Extract the features

	int width = magnitude.getWidth();
	int height = magnitude.getHeight();

	// How many times can we divide the rows and columns (considering the 50% overlap)
	int blockCountX = ((width - HOG_BLOCK_SIZE)/(HOG_CELL_SIZE)) + 1;
	int blockCountY = ((height - HOG_BLOCK_SIZE)/(HOG_CELL_SIZE)) + 1;

	gridDim.x = blockCountX;
	gridDim.y = blockCountY;
	blockDim.x = blockDim.y = HOG_BLOCK_SIZE;

	CudaMemory<float> features(blockCountX * blockCountY * HOG_NUM_CELLS * HOG_NUM_BINS, 1, 1);

	CHECK_CUDA_KERNEL((
			extractFeatures<<<gridDim, blockDim>>>
			(magnitude.getDeviceData(), orientation.getDeviceData(), magnitude.getDevicePitchInElements(), features.getMutableDeviceData())
			));


	return features;
}

CudaMemory<Npp32f> codefull::ImageUtils::getCentralDifferenceKernelX() {
	Npp32f mask[] = {1, 0, -1};
	return CudaMemory<Npp32f>(mask, 3,1,1,false, true, false);
}

CudaMemory<Npp32f> codefull::ImageUtils::getCentralDifferenceKernelY() {
	Npp32f mask[] = {1, 0, -1};
	return CudaMemory<Npp32f>(mask, 1,3,1,false, true, false);
}

int codefull::ImageUtils::getNumberOfHogFeatures(int width, int height) {
	int blockCountX = ((width - HOG_BLOCK_SIZE)/(HOG_CELL_SIZE)) + 1;
	int blockCountY = ((height - HOG_BLOCK_SIZE)/(HOG_CELL_SIZE)) + 1;
	return blockCountX * blockCountY * HOG_NUM_CELLS * HOG_NUM_BINS;
}
