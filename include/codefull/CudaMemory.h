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
	enum HeadStatus {UNINITIALIZED, CPU, GPU, SYNCED};

	HeadStatus head = UNINITIALIZED;

	T* hostData = nullptr;
	T* deviceData = nullptr;

	bool ownHostData = true;
	bool ownDeviceData = true;

	size_t width = 0;
	size_t height = 0;
	size_t numChannels = 0;
	size_t devPitch = 0;	//FIXME is this correct? pIE = pitch/(#elems * sizeof)

	/** Flag indicating whether the allocation should be pitched if possible */
	bool shouldPitch = true;

	/**
	 * @return Returns true if this memory allocation should be pitched according to
	 * 		   CUDA's requirements, false otherwise.
	 */
	virtual bool isPitched() {
		int recordSize = numChannels * sizeof(T);
		return shouldPitch && (recordSize == 4 || recordSize == 8 || recordSize == 16);
	}

	/**
	 * Transfers the data to host. If not allocated, will allocate the necessary
	 * memories
	 */
	void toHost() {
		switch(this->head) {
		case UNINITIALIZED:
			this->hostData = new T[size()];
			this->head = CPU;
			std::memset((void*) this->hostData, 0, getSizeInBytes());
			break;

		case GPU:
			if (hostData == nullptr)
				this->hostData = new T[size()];
			if (isPitched()) {	// If pitched memory is used, cudaMemcpy2d must be called to copy back the array
				CHECK_CUDA_API(cudaMemcpy2D(this->hostData, getHostPitch(),
						this->deviceData, getDevicePitch(), width * numChannels * sizeof(T), height,cudaMemcpyDeviceToHost));
			}
			else {
				CHECK_CUDA_API(cudaMemcpy((void*)this->hostData, (void*)this->deviceData, getSizeInBytes(), cudaMemcpyDeviceToHost));
			}
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
				this->devPitch = getHostPitch();
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
					this->devPitch = getHostPitch();
				}
			}

			if (isPitched()){
				CHECK_CUDA_API(cudaMemcpy2D(this->deviceData, getDevicePitch(),
						this->hostData, getHostPitch(), width * numChannels * sizeof(T), height,cudaMemcpyHostToDevice));
			}
			else {
				CHECK_CUDA_API(cudaMemcpy((void*)deviceData, (const void*)this->hostData, getSizeInBytes(), cudaMemcpyHostToDevice));
			}

			this->head = SYNCED;
			break;

		case GPU:
		case SYNCED:
			break;
		}
	}

	/**
	 * Copies all of the fields of another CudaMemory instance
	 * to this instace.
	 *
	 * @param other		Other CudaMemory instance to copy fields from
	 */
	void copyAllFields(const CudaMemory& other) {
		this->width = other.width;
		this->height = other.height;
		this->numChannels = other.numChannels;
		this->devPitch = other.devPitch;
		this->head = other.head;
		this->hostData = other.hostData;
		this->deviceData = other.deviceData;
		this->ownHostData = other.ownHostData;
		this->ownDeviceData = other.ownDeviceData;
		this->shouldPitch = other.shouldPitch;
	}

	/**
	 * Copies the information from another instance of this object to
	 * the current instance. Used by the copy constructor and the assignment constructor.
	 * @param other		Other instance to clone from
	 */
	void clone(const CudaMemory& other) {
//		this->width = other.width;
//		this->height = other.height;
//		this->numChannels = other.numChannels;
//		this->ownHostData = other.ownHostData;
//		this->ownDeviceData = other.ownDeviceData;
//		this->devPitch = other.devPitch;
		copyAllFields(other);

		switch (other.head)
		{
		case CPU:
			copyHostDataFrom(other.hostData);
			this->head = CPU;
			break;

		case GPU:
			copyDeviceDataFrom(other.deviceData);
			this->head = GPU;
			break;

		case SYNCED:
			copyHostDataFrom(other.hostData);
			toDevice();
			this->head = SYNCED;
			break;

		case UNINITIALIZED:
			this->head = UNINITIALIZED;
			break;
		}
	}


public:
	CudaMemory(int width, int height, int numChannels, bool shouldPitch = true):numChannels(numChannels),
		width(width), height(height), hostData(nullptr), deviceData(nullptr),
		ownHostData(false), ownDeviceData(false), head(UNINITIALIZED), devPitch(0), shouldPitch(shouldPitch){
	}

	CudaMemory(T* hostData, int width, int height, int numChannels, bool ownsHostData, bool clone, bool shouldPitch = true)
		   :CudaMemory(width, height, numChannels, shouldPitch) {
		setHostData(hostData, ownsHostData, clone);
	}

	CudaMemory(T* deviceData, int width, int height, int numChannels, int devicePitch, bool ownsDeviceData, bool shouldPitch = true)
		   :CudaMemory(width, height, numChannels, shouldPitch) {
		setDeviceData(deviceData, devicePitch, ownsDeviceData);
	}

	/**
	 * Copy constructor
	 */
	CudaMemory(const CudaMemory& other) {
		clone(other);
	}

	/**
	 * Move constructor
	 */
	CudaMemory(CudaMemory && other){
		copyAllFields(other);

		other.deviceData = nullptr;
		other.hostData = nullptr;
	}

	/**
	* Assignment overload
	*/
	CudaMemory& operator=(const CudaMemory& other) {
		if (this == &other)
			return *this;

		clone(other);
		return *this;
	}

	/**
	 * Move assignment overload
	 */
	CudaMemory& operator=(CudaMemory&& other) {
		if (this == &other)
			return *this;

		copyAllFields(other);
		other.deviceData = nullptr;
		other.hostData = nullptr;

		return *this;
	}

	/**
	 * Frees the allocated memories if this instance owned those
	 * data and the memory was allocated in the first place.
	 */
	virtual ~CudaMemory() {
		if (this->ownHostData && this->hostData != nullptr) {
			delete[] this->hostData;
			this->hostData = nullptr;
		}

		if (this->ownDeviceData && this->deviceData != nullptr) {
			CHECK_CUDA_API(cudaFree(this->deviceData));
			this->deviceData = nullptr;
		}
	}

	/**
	 * @return	The pitch that was assigned to this object by the CUDA API
	 */
	size_t getDevicePitch() const {
		if (this->devPitch == 0)
			throw std::runtime_error("Device memory not initialized, yet the \"pitch\" value was requested!"
					"\" (file \"" + std::string(__FILE__) +
					"\" line \"" + std::to_string(__LINE__) + "\")"
					);
		return this->devPitch;
	}

	/**
	 * @return	The pitch of the device memory in number of bytes/sizeof(T)
	 * 			Note that CUDA returns the pitch in BYTES but for calling kernels
	 * 			we are better off using pitch in elements. Otherwise, the pointer
	 * 			has to be casted to a char* pointer in the device code
	 * 			(Like shown in the CUDA C Programming guide:)
	 * 			float* row = (float*)((char*)devPtr + r * pitch);
	 */
	size_t getDevicePitchInElements() const { return this->getDevicePitch() / sizeof(T);}

	/**
	 * @return	The pitch of the host memory in number of elements (bytes/sizeof(T))
	 */
	size_t getHostPitchInElements() const {return getHostPitch() / sizeof(T);}

	/**
	 * @return	The pitch of the host memory of this instance
	 */
	size_t getHostPitch() const {return this->width * this->numChannels * sizeof(T);}

	/**
	 * @return	The height of the allocated array
	 */
	size_t getHeight() const {return this->height;}

	/**
	 * @return	The number of channels per each element of this instance
	 */
	size_t getNumChannels() const {return this->numChannels;}

	/**
	 * @return	The width of the allocated array
	 */
	size_t getWidth() const {return this->width;};

	/**
	 * @return	The size of the allocated memory in bytes
	 */
	size_t getSizeInBytes() const {return size() * sizeof(T);}

	/**
	 * @return	The size of the allocated memory in number of elements
	 */
	size_t size() const {return width * height * numChannels;}

	/**
	 * @return The unmodifiable host data (will synchronize the memory if necessary)
	 */
	const T* getHostData() {
		toHost();
		return (const T*) hostData;
	}

	/**
	 * @return The unmodifiable device data (will synchronize the memory if necessary)
	 */
	const T* getDeviceData() {
		toDevice();
		return (const T*) deviceData;
	}

	/**
	 * @return The modifiable host data (will synchronize the memory if necessary)
	 */
	T* getMutableHostData() {
		toHost();
		this->head = CPU;
		return this->hostData;
	}

	/**
	 * @return The modifiable device data (will synchronize the memory if necessary)
	 */
	T* getMutableDeviceData() {
		toDevice();
		this->head = GPU;
		return this->deviceData;
	}

	/**
	 * Sets the host array of this instance. The array will not be owned
	 * @param hostData	The host array to set as the data of this object
	 */
	void setHostData(T* hostData) {
		setHostData(hostData, false, false);
	}

	/**
	 * Sets the host array of this instance and specify whether the data
	 * is owned by this object (for memory deletion purposes)
	 * @param hostData	The host array to set as the data of this object
	 * @param ownHostData	Flag indicating whether the data is owned by
	 * 						this object
	 * @param clone		Flag indicating whether the data should be cloned
	 */
	void setHostData(T* hostData, bool ownHostData, bool clone) {
		if (this->ownHostData && this->hostData != nullptr) {
			delete[] this->hostData;
		}

		if (!clone) {
			this->hostData = hostData;
			this->ownHostData = ownHostData;
		}
		else {
			copyHostDataFrom(hostData);
		}

		this->head = CPU;
	}

	/**
	 * Sets the device array of this instance. The array will not be owned
	 * @param deviceData	The device array to set as the data of this object
	 * @param devicePitch	The pitch of the allocated memory
	 */
	void setDeviceData(T* deviceData, int devicePitch) {
			setDeviceData(deviceData, devPitch, false);
	}

	/**
	 * Sets the device array of this instance and specify whether the data
	 * is owned by this object (for memory deletion purposes)
	 * @param deviceData	The device array to set as the data of this object
	 * @param devicePitch	The pitch of the allocated memory
	 * @param ownDeviceData	Flag indicating whether the data is owned by
	 * 						this object
	 */
	void setDeviceData(T* deviceData, int devicePitch, bool ownDeviceData) {
		if (this->ownDeviceData && this->deviceData != nullptr)
			CHECK_CUDA_API(cudaFree(this->deviceData));

		this->deviceData = deviceData;
		this->ownDeviceData = ownDeviceData;
		this->devPitch = devicePitch;

		this->head = GPU;
	}

	/**
	 * Copies host data from the provided array and sets it as its own
	 * host data. Note that the width, height and the number of channels
	 * of the provided array must agree with those of this instance of
	 * CudaMemory. The assumption is that they agree and the function
	 * copies that many bytes from the source.
	 *
	 * @param hostData	The host data array to copy from
	 */
	void copyHostDataFrom(const T* hostData) {
		T* myHostData = new T[size()];

		memcpy((void*)myHostData, (const void*)hostData, getSizeInBytes());

		this->hostData = myHostData;
		this->ownHostData = true;
		this->head = CPU;
	}

	/**
	 * Copies device data from the provided array and sets it as its own
	 * device data. Note that the width, height and the number of channels
	 * of the provided array must agree with those of this instance of
	 * CudaMemory. The assumption is that they agree and the function
	 * copies that many bytes from the source.
	 *
	 * @param otherDeviceData	The device data array to copy from
	 */
	void copyDeviceDataFrom(const T* otherDeviceData) {

		if (isPitched()) {
			CHECK_CUDA_API(cudaMallocPitch((void**)&deviceData, (size_t *)&devPitch, width * numChannels * sizeof(T), height));
			CHECK_CUDA_API(cudaMemset2D((void*)this->deviceData, devPitch, 0, width * numChannels * sizeof(T), height));
		}
		else {
			CHECK_CUDA_API(cudaMalloc((void**)&this->deviceData, getSizeInBytes()));
			this->devPitch = getHostPitch();
			CHECK_CUDA_API(cudaMemset((void*) this->deviceData, 0, getSizeInBytes()));
		}

		this->head = GPU;

		if (isPitched())	// If pitched memory is used, cudaMemcpy2d must be called to copy back the array
			CHECK_CUDA_API(cudaMemcpy2D(this->deviceData, devPitch,
			otherDeviceData, devPitch, width * numChannels * sizeof(T), height, cudaMemcpyDeviceToDevice));
		else
			CHECK_CUDA_API(cudaMemcpy((void*)this->deviceData, (const void*)otherDeviceData, getSizeInBytes(), cudaMemcpyDeviceToDevice));

		this->ownDeviceData = true;
	}

	/**
	 * Clones the host data and returns it as a smart pointer. The deleter
	 * for the smart pointer is also specified.
	 *
	 * @return	The cloned host data as a smart pointer
	 */
	virtual std::shared_ptr<T> cloneHostData() {
		toHost();

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
