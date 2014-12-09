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

#include <npp.h>

#define cimg_use_jpeg
#define cimg_use_png
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
	enum Sources {FILE, MATLAB};

protected:
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
	cimg_library::CImg<float> permute(cimg_library::CImg<float> &img);

	/**
	 * Permutes the passed image back to the correct format based on the
	 * source of the image. Converts the image to the original format
	 * that the image was in.
	 *
	 * @param img	The source image (call by reference)
	 * @return The reverse-permuted image
	 */
	cimg_library::CImg<float> permuteBack(cimg_library::CImg<float> &img);

	/**
	 * Creates a CImg<float> instance using the underlying byte data.
	 * The resulting instance is permuted to the correct format but
	 * is normalized between (0,1).
	 */
	cimg_library::CImg<float> createCImg();

public:
//	using CudaMemory::CudaMemory;	// Inherit the constructors
	inline bool isPitched(){return true;}	//FIXME how to tell if really pitched??
	Sources getSource(){return this->source;}
	FloatImage(std::string filename);
	FloatImage(const float* inputImage, int width, int height, int numChannels, bool ownHostData, Sources source);
	FloatImage(float* deviceData, int width, int height, int numChannels, int devicePitch, bool ownDeviceData, Sources source);

	void display();
	void save(const std::string filename);

	std::shared_ptr<float> cloneHostData();

	std::pair<float*, int> _convolve(const Npp32f* mask, const NppiSize maskSize, const NppiPoint anchor);
	FloatImage convolve(const std::vector<Npp32f> mask, const NppiSize maskSize, const NppiPoint anchor);
	FloatImage convolve(const std::vector<Npp32f> mask, const NppiSize maskSize);
	void convolveInPlace(const std::vector<Npp32f> mask, const NppiSize maskSize, const NppiPoint anchor);
	void convolveInPlace(const std::vector<Npp32f> mask, const NppiSize maskSize);
};

} /* namespace codefull */
#endif /* FLOATIMAGE_H_ */
