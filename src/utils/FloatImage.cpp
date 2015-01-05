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

FloatImage::FloatImage(string filename) : CudaMemory(0, 0, 0) {
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

	copyHostDataFrom(image._data);
}

FloatImage::FloatImage(const float* inputImage, int width, int height, int numChannels,
		bool ownHostData, Sources source):CudaMemory(width, height, numChannels), source(source) {

	//IF FROM MATLAB, PASS THE CORRECT WIDTH AND HEIGHT! NOT WHATEVER CRAP MATLAB HAS PASSED!
	CImg<float> image = FloatImage::createCImgForSource(inputImage, width, height, numChannels, ownHostData, source);
	image.normalize(0, 1);
	permute(image);

	if (numChannels != 1 && numChannels != 3)
		throw std::runtime_error("Unsupported numchannels!");

	copyHostDataFrom(image._data);
}

FloatImage::FloatImage(float* deviceData, int width, int height, int numChannels, int devicePitch,
		bool ownDeviceData, Sources source):CudaMemory(width, height, numChannels){
	this->source = source;
	setDeviceData(deviceData, devicePitch, ownDeviceData);
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

CImg<float>& FloatImage::permute(CImg<float>& img) {
	return permute(img, this->source);
}

CImg<float>& FloatImage::permute(CImg<float>& img, Sources target) {
	toHost();

	switch (target) {
	case MATLAB:
		return img.permute_axes("cyxz");

	case FILE:
		return img.permute_axes("cxyz");
		break;

	case DATA_PTR:
		return img;
		break;

	default:
		return img;
	}
}

CImg<float>& FloatImage::permuteBack(CImg<float>& img) {
	return permuteBack(img, this->source);
}

CImg<float>& FloatImage::permuteBack(CImg<float>& img, Sources target) {
	toHost();

	switch (target) {
	case MATLAB:
		img.permute_axes("zycx");
		return img;

	case FILE:
		return img.permute_axes("yzcx");

	case DATA_PTR:
		return img;

	default:
		return img;
	}
}

cimg_library::CImg<float> FloatImage::toCImg() {
	return this->toCImg(this->source);
}

cimg_library::CImg<float> FloatImage::toCImg(Sources target) {
	toHost();

	switch(target) {
	case MATLAB:{
		CImg<float> result(getHostData(), numChannels, width, height, 1, false);
		return permuteBack(result, target);
	}
	case FILE: {
		CImg<float> result(getHostData(), numChannels, width, height, 1, false);
		return permuteBack(result, target);
	}

	case DATA_PTR:
	default: {
		CImg<float> result(getHostData(), width, height, 1, numChannels, false);
		return result;
	}
	}
}

void FloatImage::display() {
	CImg<float> res = toCImg(FILE);
	res.normalize(0, 255);
	res.display();
}

void FloatImage::save(const std::string filename) {
	CImg<float> res = toCImg(FILE);
	res.normalize(0, 255);
	res.save(filename.c_str());
}

std::shared_ptr<float> FloatImage::toByteArray(Sources target) {
	float* finalData = toCImg(target).data();

	// Clone the data into a new array
	float* result = new float[size()];
	memcpy((void*) result, (const void*) finalData, getSizeInBytes());
	std::shared_ptr<float> smartP( result, []( float *p ) { delete[] p; } );
	return smartP;
}

void FloatImage::clone(const FloatImage& other) {
	CudaMemory<float>::clone(other);
	this->source = other.source;
}

void FloatImage::copyAllFields(const FloatImage& other) {
	CudaMemory<float>::copyAllFields(other);
	this->source = other.source;
}

std::pair<float*, int> FloatImage::_convolve(CudaMemory<Npp32f> mask, const NppiPoint anchor) {
	const float* devMask = mask.getDeviceData();

	// Allocate target memory
	int devDstPitch;
	float* devDst = getNppAllocator()(width, height, &devDstPitch);

	// Define and set parameters
	NppiSize sourceSize = {(int)getWidth(), (int)getHeight()};
	NppiPoint sourceOffset = {0, 0};
	NppiSize roi = NppiSize{(int)getWidth(), (int)getHeight()};

	const float* devSrc = getDeviceData();
	int devSrcPitch = getDevicePitch();

	// Do filter
	CHECK_NPP_API(getNppConvolution()(devSrc, devSrcPitch, sourceSize, sourceOffset,
				devDst, devDstPitch, roi,
				devMask, NppiSize{(int)mask.getWidth(), (int)mask.getHeight()}, anchor, NPP_BORDER_REPLICATE));


	return pair<float*, int> (devDst, devDstPitch);
}

FloatImage FloatImage::convolve(CudaMemory<Npp32f> mask, const NppiPoint anchor) {
	pair<float*, int> convolved = this->_convolve(mask, anchor);
	return FloatImage(convolved.first, width, height, numChannels, convolved.second, true, Sources::DATA_PTR);
}

FloatImage FloatImage::convolve(CudaMemory<Npp32f> mask) {
	NppiPoint defaultAnchor = {(int)mask.getWidth() / 2, (int)mask.getHeight() / 2};
	return this->convolve(mask, defaultAnchor);
}

void FloatImage::convolveInPlace(CudaMemory<Npp32f> mask, const NppiPoint anchor) {
	pair<float*, int> convolved = this->_convolve(mask, anchor);
	setDeviceData(convolved.first, convolved.second, true);
}

void FloatImage::convolveInPlace(CudaMemory<Npp32f> mask) {
	NppiPoint defaultAnchor = {(int)mask.getWidth() / 2, (int)mask.getHeight() / 2};
	convolveInPlace(mask, defaultAnchor);
}

} /* namespace codefull */
