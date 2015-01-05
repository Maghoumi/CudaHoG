/*
 * FloatImage.h
 *
 *  Created on: Nov 2, 2014
 *      Author: Mehran Maghoumi
 *
 *  Implements a floating point image container that can be safely passed to and from the device
 */

#ifndef FLOATIMAGE_H_
#define FLOATIMAGE_H_

#include <fstream>
#include <string>
#include <stdexcept>
#include <cstring>
#include <vector>
#include <memory>

#include <npp.h>

#ifndef _MATLAB		// MATLAB goes apeshit with imagemagick
	#define cimg_use_magick	// For CIMG
#endif

#include <CImg.h>

#include "commons.h"

#include "CudaMemory.h"

namespace codefull {

class FloatImage : public CudaMemory<float> {

	/**
	 * Typedefs for function pointers
	 */
	typedef Npp32f* (*NppAllocator)(int, int, int*);
	typedef NppStatus (*NppConvolution) (const Npp32f * pSrc, Npp32s nSrcStep,NppiSize oSrcSize, NppiPoint oSrcOffset,
										 Npp32f * pDst, Npp32s nDstStep,
										 NppiSize oSizeROI,
										 const Npp32f * pMask, NppiSize oMaskSize, NppiPoint oAnchor,
										 NppiBorderType eBorderType);

public:
	/**
	 * Defines the possible sources of this image. The source
	 * is important as it will determine the order of the axes
	 */
	enum Sources {FILE, MATLAB, DATA_PTR};

protected:
	void clone(const FloatImage& other);
	void copyAllFields(const FloatImage& other);

	/**
	 * Returns an NPP memory allocator function based on the
	 * number of channels of this image
	 */
	NppAllocator getNppAllocator();

	/**
	 * Returns an NPP convolution filter function based
	 * on the number of the channels in this image
	 */
	NppConvolution getNppConvolution();

	/** The source data of this image */
	Sources source = FILE;

	/**
	 * Permutes the passed image to the correct format based on the
	 * source of the image. Makes the image suitable for processing
	 * with this library
	 *
	 * @param img	The source image (call by reference)
	 * @return The permuted image
	 */
	cimg_library::CImg<float>& permute(cimg_library::CImg<float> &img);

	/**
	 * Permutes the passed image to the correct format specified as
	 * the target. Makes the image suitable for processing
	 * with this library
	 *
	 * @param img	The source image (call by reference)
	 * @param target	The target format of the image
	 * @return The permuted image
	 */
	cimg_library::CImg<float>& permute(cimg_library::CImg<float> &img, Sources target);

	/**
	 * Permutes the passed image back to the correct format based on the
	 * source of the image. Converts the image to the original format
	 * that the image was in.
	 *
	 * @param img	The source image (call by reference)
	 * @return The reverse-permuted image
	 */
	cimg_library::CImg<float>& permuteBack(cimg_library::CImg<float> &img);

	/**
	 * Permutes the passed image back to the format specified as the
	 * target. Converts the image to the original format
	 * that the image was in.
	 *
	 * @param img	The source image (call by reference)
	 * @param target	The target format of the image
	 * @return The reverse-permuted image
	 */
	cimg_library::CImg<float>& permuteBack(cimg_library::CImg<float> &img, Sources target);

	/**
	 * Converts this FloatImage to a proper CImg<float> instance based
	 * on the original type of the data that was used to create this
	 * instance of the class.
	 *
	 * Creates a CImg<float> instance using the underlying byte data.
	 * The resulting instance is permuted to the correct format but
	 * is normalized between (0,1).
	 */
	cimg_library::CImg<float> toCImg();

	/**
	 * Converts this FloatImage to a proper CImg<float> instance based
	 * on the target type requested (eg. file, MATLAB, etc)
	 *
	 * Creates a CImg<float> instance using the underlying byte data.
	 * The resulting instance is permuted to the correct format but
	 * is normalized between (0,1).
	 *
	 * @param	target	The target format of the CImg instance
	 * 					(eg. transposed for MATLAB or planer for file etc.)
	 */
	cimg_library::CImg<float> toCImg(Sources target);

	/**
	 * Utility method to instantiate a new CImg<float> object for the given source.
	 * @param inputImage	Array of image data
	 * @param width		The real width of the image (MATLAB's second array dimension must be provided)
	 * @param height	The real height of the image (MATLAB's first array dimension must be provided)
	 * @param numChannels	The real number of color channels of the image (MATLAB's third array dimension must be provided)
	 */
	static cimg_library::CImg<float> createCImgForSource(const float* inputImage, int width, int height, int numChannels,
			bool ownHostData, Sources source) {
		switch(source) {
		case MATLAB:
			return cimg_library::CImg<float> (inputImage, height, width, 1, numChannels, ownHostData);

		case FILE:
		case DATA_PTR:
		default:
			return cimg_library::CImg<float> (inputImage, width, height, 1, numChannels, ownHostData);
		}
	}

public:
	explicit FloatImage(std::string filename);
	FloatImage(const float* inputImage, int width, int height, int numChannels, bool ownHostData, Sources source);
	FloatImage(float* deviceData, int width, int height, int numChannels, int devicePitch, bool ownDeviceData, Sources source);

	//TODO test move operations

	/**
	 * Copy constructor
	 */
	FloatImage(const FloatImage& other):CudaMemory(other.width, other.height, other.numChannels) {
		clone(other);
	}

	/**
	 * Move constructor
	 */
	FloatImage(FloatImage && other):CudaMemory(other) {
		copyAllFields(other);
		other.deviceData = nullptr;
		other.hostData = nullptr;
	}

	/**
	 * Overloaded assignment
	 */
	FloatImage& operator=(const FloatImage& other) {
		if (this == &other)
			return *this;

		clone(other);
		return *this;
	}

	/**
	 * Overloaded move assignment
	 */
	FloatImage& operator=(FloatImage&& other) {
		if (this == &other)
			return *this;

		copyAllFields(other);
		other.deviceData = nullptr;
		other.hostData = nullptr;

		return *this;
	}
	
	inline bool isPitched(){ return true; }	//FIXME how to tell if really pitched??
	Sources getSource(){ return this->source; }

	void display();
	void save(const std::string filename);

	virtual std::shared_ptr<float> toByteArray(Sources target);

	std::pair<float*, int> _convolve(CudaMemory<Npp32f> mask, const NppiPoint anchor);
	FloatImage convolve(CudaMemory<Npp32f> mask, const NppiPoint anchor);
	FloatImage convolve(CudaMemory<Npp32f> mask);
	void convolveInPlace(CudaMemory<Npp32f> mask, const NppiPoint anchor);
	void convolveInPlace(CudaMemory<Npp32f> mask);
};

} /* namespace codefull */
#endif /* FLOATIMAGE_H_ */
