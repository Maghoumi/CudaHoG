/*
 * ImageUtils.cuh
 *
 *  Created on: Oct 24, 2014
 *      Author: Mehran Maghoumi
 *
 *  Provides some utilities that are required for image processing
 *  tasks
 */

#ifndef IMAGEUTILS_H_
#define IMAGEUTILS_H_

#include <npp.h>
#include <memory>
#include <vector>

#include "commons.h"
#include "CudaMemory.h"
#include "FloatImage.h"

namespace codefull {

class ImageUtils {
public:
	enum KernelDirection {DIRECTION_X, DIRECTION_Y, DIRECTION_BOTH};

	/**
	 * Generates the central difference kernel for convolution
	 * {-1, 0, 1}
	 *
	 * The kernel is unidirectional and the user is responsible
	 * for providing the correct kernel size to Image object for
	 * convolution.
	 *
	 * NOTE: Coefficients are reversed (NPP requirement)
	 */
	static std::vector<Npp32s> getCentralDifferenceKernel();

	static std::vector<Npp32f> getCentralDifferenceKernelFloat();

	static std::vector<Npp32f> getBoxKernel(const int size);

	/**
	 * Extracts the HoG features from the given FloatImage instance.
	 * The cell size is 8, block size is 16 and blocks have 50% overlap.
	 * The blocks are normalized using the L2-norm
	 */
	static CudaMemory<float> extractHOGFeatures(FloatImage &inputImage);

	/**
	 * Returns the length of the HoG feature vector for the specified
	 * width and height.
	 *
	 * @param width		The height of the image
	 * @param height	The width of the image
	 *
	 * @return The length of the HoG feature vector for cell size of 8
	 * 		   and block size of 16
	 */
	static int getNumberOfHogFeatures(int width, int height);
};
}

#endif /* IMAGEUTILS_H_ */
