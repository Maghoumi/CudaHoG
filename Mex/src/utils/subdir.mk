################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/utils/FloatImage.cpp \
../src/utils/PreciseTimer.cpp 

CU_SRCS += \
../src/utils/ImageUtils.cu 

CU_DEPS += \
./src/utils/ImageUtils.d 

OBJS += \
./src/utils/FloatImage.o \
./src/utils/ImageUtils.o \
./src/utils/PreciseTimer.o 

CPP_DEPS += \
./src/utils/FloatImage.d \
./src/utils/PreciseTimer.d 


# Each subdirectory must supply rules for building sources it contributes
src/utils/%.o: ../src/utils/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-6.5/bin/nvcc -I"/media/mehran/Cloud/University Stuff/UCF/CudaHoG" -G -g -pg -O0 -Xcompiler -fPIC -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 --target-cpu-architecture x86 -m64 -odir "src/utils" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-6.5/bin/nvcc -I"/media/mehran/Cloud/University Stuff/UCF/CudaHoG" -G -g -pg -O0 -Xcompiler -fPIC --compile --target-cpu-architecture x86 -m64  -x c++ -o  "$@" -c "$<" -std=c++11
	@echo 'Finished building: $<'
	@echo ' '

src/utils/%.o: ../src/utils/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-6.5/bin/nvcc -I"/media/mehran/Cloud/University Stuff/UCF/CudaHoG" -G -g -pg -O0 -Xcompiler -fPIC -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 --target-cpu-architecture x86 -m64 -odir "src/utils" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-6.5/bin/nvcc -I"/media/mehran/Cloud/University Stuff/UCF/CudaHoG" -G -g -pg -O0 -Xcompiler -fPIC --compile --relocatable-device-code=true -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_21 --target-cpu-architecture x86 -m64  -x cu -o  "$@" -c "$<" -std=c++11
	@echo 'Finished building: $<'
	@echo ' '


