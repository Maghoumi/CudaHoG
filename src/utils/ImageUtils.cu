/*
 * ImageUtils.cu
 *
 *  Created on: Oct 24, 2014
 *      Author: Mehran Maghoumi
 *
 */

#include "ImageUtils.cuh"
#include "ImageUtilsKernels.cuh"

using namespace std;

std::vector<Npp32s> codefull::ImageUtils::getCentralDifferenceKernel() {
	std::vector<Npp32s> result = {1, 0, -1};
	return result;
}

std::vector<Npp32f> codefull::ImageUtils::getCentralDifferenceKernelFloat() {
	std::vector<Npp32f> result = {1, 0, -1};
	return result;
}

std::vector<Npp32f> codefull::ImageUtils::getBoxKernel(const int size) {
	std::vector<Npp32f> result;
	float size2 = size * size;

	for (int i = 0 ; i < size * size ; i++) {
		result.push_back(1 / size2);
	}

	return result;
}

codefull::CudaMemory<float> codefull::ImageUtils::extractHOGFeatures(codefull::FloatImage &image) {
	// Step 1: Find the image gradients
	vector<Npp32f> mask = ImageUtils::getCentralDifferenceKernelFloat();

	/**
	 * According to MATLAB's output:
	 * 		{3,1} gives you gradX
	 * 		{1,3} gives you gradY
	 * 	The transposed cases are handled internally by _convolve ;)
	 * 	However, the sizes must be swapped as MATLAB is doing some
	 * 	ugly transposition under the hood :(
	 */
	NppiSize maskSizeX;
	NppiSize maskSizeY;

	// If the source of the image was MATLAB, the angles must be swapped
	if (false && image.getSource() == FloatImage::MATLAB) {
		maskSizeX = {1, 3};
		maskSizeY = {3, 1};
	}
	else {
		maskSizeX = {3, 1};
		maskSizeY = {1, 3};
	}

	FloatImage gradX = image.convolve(mask, maskSizeX);
	FloatImage gradY = image.convolve(mask, maskSizeY);

	CudaMemory<float> magnitude(image.getWidth(), image.getHeight(), 1);magnitude.getMutableDeviceData();
	CudaMemory<float> orientation(image.getWidth(), image.getHeight(), 1);orientation.getMutableDeviceData();

	dim3 gridDim(DIVUP(image.getWidth(), HOG_MAGNITUDE_BLOCK_SIZE), DIVUP(image.getHeight(), HOG_MAGNITUDE_BLOCK_SIZE), 1);
	dim3 blockDim(HOG_MAGNITUDE_BLOCK_SIZE, HOG_MAGNITUDE_BLOCK_SIZE, 1);


	CHECK_CUDA_KERNEL((
	calculateMagnitudeAndOrientation<<<gridDim, blockDim>>>(gradX.getDeviceData(), gradY.getDeviceData(), gradX.getDevicePitch()/4,
			 magnitude.getMutableDeviceData(), orientation.getMutableDeviceData(), magnitude.getDevicePitch()/4,
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
			(magnitude.getDeviceData(), orientation.getDeviceData(), magnitude.getDevicePitch()/4, features.getMutableDeviceData())
			));

	return features;
}

int codefull::ImageUtils::getNumberOfHogFeatures(int width, int height) {
	int blockCountX = ((width - HOG_BLOCK_SIZE)/(HOG_CELL_SIZE)) + 1;
	int blockCountY = ((height - HOG_BLOCK_SIZE)/(HOG_CELL_SIZE)) + 1;
	return blockCountX * blockCountY * HOG_NUM_CELLS * HOG_NUM_BINS;
}
