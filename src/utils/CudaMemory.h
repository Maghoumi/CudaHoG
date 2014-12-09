/*
 * CudaMemory.h
 *
 *  Created on: Oct 25, 2014
 *      Author: Mehran Maghoumi
 *
 *  Defines a memory that is synchronized between the host and device
 *  on demand.
 */

#ifndef CUDAMEMORY_H_
#define CUDAMEMORY_H_

#include <cstring>
#include <memory>

#include <cuda.h>


#include "commons.h"

namespace codefull {

template <class T>
class CudaMemory {

protected:
	CudaMemory(){}	//FIXME for inheritance purposes
	enum HeadStatus {UNINITIALIZED, CPU, GPU, SYNCED};

	HeadStatus head = UNINITIALIZED;

	T* hostData = nullptr;
	T* deviceData = nullptr;

	bool ownHostData = true;
	bool ownDeviceData = true;
	bool transposed = false;

	unsigned int width = 0;
	unsigned int height = 0;
	unsigned int numChannels = 0;
	long unsigned int devPitch = 0;	//FIXME is this correct? pIE = pitch/(#elems * sizeof)

	/**
	 * @return Returns true if this memory allocation should be pitched according to
	 * 		   CUDA's requirements, false otherwise.
	 */
	virtual inline bool isPitched() {
		int recordSize = numChannels * sizeof(T);
		return recordSize == 4 || recordSize == 8 || recordSize == 16;
	}

	/**
	 * Transfers the data to host. If not allocated, will allocate the necessary
	 * memories
	 */
	void toHost() {
		switch(this->head) {
		case UNINITIALIZED:
			this->hostData = new T[getSizeInBytes()];
			this->head = CPU;
			std::memset((void*) this->hostData, 0, getSizeInBytes());
			break;

		case GPU:
			if (hostData == nullptr)
				this->hostData = new T[getSizeInBytes()];
			if (isPitched())	// If pitched memory is used, cudaMemcpy2d must be called to copy back the array
				CHECK_CUDA_API(cudaMemcpy2D(this->hostData, getHostPitch(),
						this->deviceData, getDevicePitch(), width * numChannels * sizeof(T), height,cudaMemcpyDeviceToHost));
			else
				CHECK_CUDA_API(cudaMemcpy((void*)this->hostData, (void*)this->deviceData, getSizeInBytes(), cudaMemcpyDeviceToHost));
			this->head = SYNCED;
			break;

		case CPU:
		case SYNCED:
			break;
		}
	}

	/**
	 * Transfers the data to device. If not allocated, will allocate the necessary
	 * memories.
	 */
	void toDevice() {
		switch (this->head) {
		case UNINITIALIZED:
			if (isPitched()) {
				CHECK_CUDA_API(cudaMallocPitch((void**)&deviceData, &devPitch, width * numChannels * sizeof(T), height));
				CHECK_CUDA_API(cudaMemset2D((void*)this->deviceData,devPitch, 0, width * numChannels * sizeof(T), height));
			}
			else {
				CHECK_CUDA_API(cudaMalloc((void**) &this->deviceData, getSizeInBytes()));
				this->devPitch = getHostPitch();	//fixme
				CHECK_CUDA_API(cudaMemset((void*) this->deviceData, 0, getSizeInBytes()));
			}
			this->head = GPU;
			break;

		case CPU:
			if (this->deviceData == nullptr) {	// Allocate if necessary
				if (isPitched()) {
					CHECK_CUDA_API(cudaMallocPitch((void**)&deviceData, &devPitch, width * numChannels * sizeof(T), height));
				}
				else {
					CHECK_CUDA_API(cudaMalloc((void**) &this->deviceData, getSizeInBytes()));
					this->devPitch = getHostPitch();	//fixme
				}
			}

			if (isPitched())
				CHECK_CUDA_API(cudaMemcpy2D(this->deviceData, getDevicePitch(),
						this->hostData, getHostPitch(), width * numChannels * sizeof(T), height,cudaMemcpyHostToDevice));
			else
				CHECK_CUDA_API(cudaMemcpy((void*)deviceData, (void*)this->hostData, getSizeInBytes(), cudaMemcpyHostToDevice));

			this->head = SYNCED;
			break;

		case GPU:
		case SYNCED:
			break;
		}
	}


public:
	CudaMemory(int width, int height, int numChannels, bool transposed = false){
		this->numChannels = numChannels;
		this->transposed = transposed;
		if (!transposed) {
			this->width = width;
			this->height = height;
		}
		else {
			this->width = height;
			this->height = width;
		}
	}

	CudaMemory(T* hostData, int width, int height, int numChannels, bool ownsHostData, bool transposed = false)
		   :CudaMemory(width, height, numChannels, transposed) {
		setHostData(hostData, ownsHostData);
	}

	CudaMemory(T* deviceData, int width, int height, int numChannels, int devicePitch, bool ownsDeviceData, bool transposed = false)
		   :CudaMemory(width, height, numChannels, transposed) {
		setDeviceData(deviceData, devicePitch, ownsDeviceData);
	}

	~CudaMemory() {
		if (this->ownHostData && this->hostData != nullptr) {
			delete[] this->hostData;
		}

		if (this->ownDeviceData && this->deviceData != nullptr)
			CHECK_CUDA_API(cudaFree(this->deviceData));

		this->hostData = nullptr;
		this->deviceData = nullptr;
	}

	inline int getDevicePitch() const {
		if (this->devPitch == 0)
			throw std::runtime_error("Device memory not initialized, yet the \"pitch\" value was requested!"
					"\" (file \"" + std::string(__FILE__) +
					"\" line \"" + std::to_string(__LINE__) + "\")"
					);
		return this->devPitch;
	}

	/**
	 * @return the pitch of the device memory in NUMBER OF ELEMENTS!
	 * Not byte! CUDA likes to return them in bytes (eg cudaMallocPitch)
	 */
	bool isTransposed() const {return this->transposed;}
	int getDevicePitchInElements() const { return this->getDevicePitch() / (this->numChannels * sizeof(T));}
	int getHeight() const {return this->height;}
	int getNumChannels() const {return this->numChannels;}
	int getHostPitchInElements() const {return getHostPitch() / (this->numChannels * sizeof(T));}
	int getHostPitch() const {return this->width * this->numChannels * sizeof(T);}
	int getWidth() const {return this->width;};
	int getSizeInBytes() const {return size() * sizeof(T);}
	int size() const {return width * height * numChannels;}

	/**
	 * @return Returns the unmodifiable host data
	 */
	const T* getHostData() {
		toHost();
		return (const T*) hostData;
	}

	void setHostData(T* hostData) {
		setHostData(hostData, false);
	}

	void setHostData(T* hostData, bool ownHostData) {
		if (this->ownHostData && this->hostData != nullptr) {
			delete[] this->hostData;
		}

		this->hostData = hostData;
		this->ownHostData = ownHostData;

		this->head = CPU;
	}

	/**
	 * Copies host data from the provided array and sets it as its own
	 * host data. Note that the width, height and the number of channels
	 * of the provided array must agree with those of this instance of
	 * CudaMemory. The assumption is that they agree and the function
	 * copies that many bytes from the source.
	 */
	void copyHostDataFrom(const T* hostData) {
		T* myHostData = new T[width * height * numChannels];

		memcpy((void*)myHostData, (const void*)hostData, width * height * numChannels * sizeof(T));

		this->hostData = myHostData;
		this->ownHostData = true;
		this->head = CPU;
	}

	void setDeviceData(T* deviceData, int devicePitch) {
		setDeviceData(deviceData, devPitch, false);
	}

	void setDeviceData(T* deviceData, int devicePitch, bool ownDeviceData) {
		if (this->ownDeviceData && this->deviceData != nullptr)
			CHECK_CUDA_API(cudaFree(this->deviceData));

		this->deviceData = deviceData;
		this->ownDeviceData = ownDeviceData;
		this->devPitch = devicePitch;

		this->head = GPU;
	}

	/**
	 * @return Returns the unmodifiable device data
	 */
	const T* getDeviceData() {
		toDevice();
		return (const T*) deviceData;
	}

	T* getMutableHostData() {
		toHost();
		this->head = CPU;
		return this->hostData;
	}

	T* getMutableDeviceData() {
		toDevice();
		this->head = GPU;
		return this->deviceData;
	}

	virtual std::shared_ptr<T> cloneHostData() {
		getHostData();

		// Clone the data into a new array
		T* result = new T[size()];
		memcpy((void*) result, (const void*) hostData, getSizeInBytes());
		std::shared_ptr<T> smartP( result, []( T *p ) { delete[] p; } );
		return smartP;
	}

};

// Force compilation to detect my errors
template class CudaMemory<int>;

} /* namespace codefull */

#endif /* CUDAMEMORY_H_ */
