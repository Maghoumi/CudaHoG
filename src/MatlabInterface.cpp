#include "MatlabInterface.h"

std::shared_ptr<float> extractHOGFeatures(float* inputImage, int dim0, int dim1, int dim2) {
	FloatImage image(inputImage, dim0, dim1, dim2, false, FloatImage::MATLAB);
	return ImageUtils::extractHOGFeatures(image).cloneHostData();
}

std::shared_ptr<float> filter (float* inputImage, int dim0, int dim1, int dim2, int filterSize) {
	FloatImage image(inputImage, dim0, dim1, dim2, false, FloatImage::MATLAB);
	return image.convolve(ImageUtils::getBoxKernel(filterSize)).toByteArray(FloatImage::Sources::MATLAB);
}
