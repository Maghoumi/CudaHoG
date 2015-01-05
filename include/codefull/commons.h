/*
 * commons.h
 *
 *  Created on: Oct 21, 2014
 *      Author: Mehran Maghoumi
 */

#ifndef COMMONS_H_
#define COMMONS_H_

#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

/**
 * CUDA API call error check macro
 */
#define CHECK_CUDA_API(S) do {cudaError_t eCUDAResult; \
        eCUDAResult = S; \
        if (eCUDAResult != cudaSuccess)\
        	throw std::runtime_error("Error calling CUDA API in function \"" + std::string(__FUNCTION__)\
        						 + "\" (file \"" + std::string(__FILE__)\
        						 + "\" line \"" + std::to_string(__LINE__) + "\")"\
        			+ "! Error: \"" + \
        			std::string(cudaGetErrorString(eCUDAResult)) + "\" error #: \"" + std::to_string(eCUDAResult) + "\""); \
        } while (false)

#define CHECK_NPP_API(S) do {NppStatus eStatusNPP; \
        eStatusNPP = S; \
        if (eStatusNPP != NPP_SUCCESS)\
        throw std::runtime_error("Error calling NPP function! Error#: " + std::to_string(eStatusNPP)); \
        } while (false)

#define CHECK_CUDA_KERNEL(S) do {cudaError_t eCUDAResult; \
		S;\
		cudaDeviceSynchronize();\
        eCUDAResult = cudaGetLastError(); \
        if (eCUDAResult != cudaSuccess)\
        	throw std::runtime_error("Error calling CUDA kernel in function \"" + std::string(__FUNCTION__)\
        						 + "\" (file \"" + std::string(__FILE__)\
        						 + "\" line \"" + std::to_string(__LINE__) + "\")"\
        			+ "! Error: \"" + \
        			std::string(cudaGetErrorString(eCUDAResult)) + "\" error #: \"" + std::to_string(eCUDAResult) + "\""); \
        } while (false)


/**
 * Macros for disabling copy and assignment operations
 */
#define DISABLE_COPY(classname) \
	private:\
	classname(const classname&)

#define DISABLE_ASSIGN(classname) \
	private:\
	classname& operator=(const classname&)

#define DISABLE_COPY_AND_ASSIGN(classname) \
private:\
  classname(const classname&);\
  classname& operator=(const classname&)

#endif /* COMMONS_H_ */
