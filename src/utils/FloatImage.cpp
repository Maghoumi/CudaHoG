/*
 * FloatImage.cpp
 *
 *  Created on: Nov 2, 2014
 *      Author: Mehran Maghoumi
 */

#include "codefull/FloatImage.h"

namespace codefull {

using namespace std;
using namespace cimg_library;

FloatImage::FloatImage(string filename):CudaMemory() {
	ifstream file(filename.data(), ifstream::in);

	// Does the file exist?
	if (!file.good()) {
		throw invalid_argument("File \"" + filename + "\" not found!");
	}

	file.close();

	CImg<float> image(filename.c_str());
	image.normalize(0, 1);	// FIXME performance?

	this->width = image.width();
	this->height = image.height();
	this->numChannels = image.spectrum();
	this->source = FILE;

	if (numChannels != 1 && numChannels != 3)
		throw std::runtime_error("Unsupported numchannels!");

	// CImg stores images as planer. The axes need to be permutated.
	// WARNING: This operation changes the dimensions of the image
	// but since we've already saved it, won't matter.
	// However, during saving or displaying, this needs to be corrected
	permute(image);
//	image.permute_axes("cxyz");

	copyHostDataFrom(image._data);
}

FloatImage::FloatImage(float* deviceData, int width, int height, int numChannels, int devicePitch, bool ownDeviceData, Sources source) {
	this->width = width;
	this->height = height;
	this->numChannels = numChannels;
	this->source = source;
	setDeviceData(deviceData, devicePitch, ownDeviceData);
}

FloatImage::FloatImage(const float* inputImage, int width, int height, int numChannels, bool ownHostData, Sources source):CudaMemory() {
	this->numChannels = numChannels;
	this->width = height;
	this->height = width;

	CImg<float> image(inputImage, width, height, 1, numChannels, ownHostData);

	image.normalize(0, 1);	// FIXME performance?
	this->source = source;

//	if (source == MATLAB) {
//		image.permute_axes("cyxz");	// In one go, transpose (for MATLAB) and make interleaved
//	}
//	else if (source == FILE) {
//		image.permute_axes("cxyz");
//	}
	permute(image);

	if (numChannels != 1 && numChannels != 3)
		throw std::runtime_error("Unsupported numchannels!");

	copyHostDataFrom(image._data);
}

FloatImage::NppAllocator FloatImage::getNppAllocator() {
	switch(getNumChannels()) {
	case 1:
		return &nppiMalloc_32f_C1;
	case 2:
		return &nppiMalloc_32f_C2;
	case 3:
		return &nppiMalloc_32f_C3;
	case 4:
		return &nppiMalloc_32f_C4;
	}

	return nullptr;
}

FloatImage::NppConvolution FloatImage::getNppConvolution() {
	switch(getNumChannels()) {
	case 1:
		return &nppiFilterBorder_32f_C1R;
	case 3:
		return &nppiFilterBorder_32f_C3R;
	case 4:
		return &nppiFilterBorder_32f_C4R;
	}
	return nullptr;
}

CImg<float> FloatImage::permute(CImg<float>& img) {
	switch (this->source) {
	case MATLAB:
		return img.permute_axes("cyxz");
		break;

	case FILE:
		return img.permute_axes("cxyz");
		break;
	}
}

CImg<float> FloatImage::permuteBack(CImg<float>& img) {
	switch (this->source) {
	case MATLAB:
		return img.permute_axes("zycx");
		break;

	case FILE:
		return img.permute_axes("yzcx");
		break;
	}
}

cimg_library::CImg<float> FloatImage::createCImg() {

	if (this->source == MATLAB) {
		CImg<float> result(getHostData(), numChannels, width, height, 1, false);
		return permuteBack(result);
	}
	else if (this->source == FILE) {
		CImg<float> result(getHostData(), numChannels, width, height, 1, false);
		return permuteBack(result);
	}
}

void FloatImage::display() {
	createCImg().normalize(0, 255).display();
}

void FloatImage::save(const std::string filename) {
	createCImg().normalize(0, 255).save(filename.c_str());
}

shared_ptr<float> FloatImage::cloneHostData() {
	CImg<float> img (getHostData(), numChannels, width, height, 1, false);

	if (this->source == MATLAB)
		img.permute_axes("zycx"); // In one go, make planer and go back to MATLAB format
	else if (this->source == FILE)
		img.permute_axes("yzcx"); // In one go, make planer and go back to MATLAB format

	float* finalData = img.data();

	// Clone the data into a new array
	float* result = new float[size()];
	memcpy((void*) result, (const void*) finalData, getSizeInBytes());
	std::shared_ptr<float> smartP( result, []( float *p ) { delete[] p; } );
	return smartP;
}

std::pair<float*, int> FloatImage::_convolve(const Npp32f* mask, const NppiSize maskSize, const NppiPoint anchor) {
	NppiSize correctedMaskSize;	// Corrected to account for transposition

	correctedMaskSize.width = maskSize.width;
	correctedMaskSize.height = maskSize.height;

	// Allocate mask
	Npp32f *devMask;


	CHECK_CUDA_API(cudaMalloc((void**)&devMask, correctedMaskSize.width * correctedMaskSize.height * sizeof(Npp32f)));
	CHECK_CUDA_API(cudaMemcpy(devMask,mask, correctedMaskSize.width * correctedMaskSize.height * sizeof(Npp32f), cudaMemcpyHostToDevice));

	// Allocate target memory
	int devDstPitch;
	float* devDst = getNppAllocator()(width, height, &devDstPitch);	//FIXME for multiple channels

	// Define and set parameters
	NppiSize sourceSize = {(int)getWidth(), (int)getHeight()};
	NppiPoint sourceOffset = {0, 0};
	NppiSize roi = NppiSize{(int)getWidth(), (int)getHeight()};

	const float* devSrc = getDeviceData();
	int devSrcPitch = getDevicePitch();

	// Do filter
	CHECK_NPP_API(getNppConvolution()(devSrc, devSrcPitch, sourceSize, sourceOffset,
				devDst, devDstPitch, roi,
				devMask, correctedMaskSize, anchor, NPP_BORDER_REPLICATE));

	CHECK_CUDA_API(cudaFree(devMask));

	return pair<float*, int> (devDst, devDstPitch);
}

FloatImage FloatImage::convolve(const std::vector<Npp32f> mask, const NppiSize maskSize, const NppiPoint anchor) {
	pair<float*, int> convolved = this->_convolve(&mask[0], maskSize, anchor);
	return FloatImage(convolved.first, width, height, numChannels, convolved.second, true, this->source);
}

FloatImage FloatImage::convolve(const std::vector<Npp32f> mask, const NppiSize maskSize) {
	NppiPoint defaultAnchor = {maskSize.width / 2, maskSize.height / 2};
	return this->convolve(mask, maskSize, defaultAnchor);
}

void FloatImage::convolveInPlace(const std::vector<Npp32f> mask, const NppiSize maskSize, const NppiPoint anchor) {
	pair<float*, int> convolved = this->_convolve(&mask[0], maskSize, anchor);
	setDeviceData(convolved.first, convolved.second, true);
}

void FloatImage::convolveInPlace(const std::vector<Npp32f> mask, const NppiSize maskSize) {
	NppiPoint defaultAnchor = {maskSize.width / 2, maskSize.height / 2};
	convolveInPlace(mask, maskSize, defaultAnchor);
}

} /* namespace codefull */
