################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/Main.cpp \
../src/MatlabInterface.cpp 

OBJS += \
./src/Main.o \
./src/MatlabInterface.o 

CPP_DEPS += \
./src/Main.d \
./src/MatlabInterface.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-6.5/bin/nvcc -I"/media/mehran/Cloud/University Stuff/UCF/CudaHoG" -G -g -pg -O0 -Xcompiler -fPIC -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 --target-cpu-architecture x86 -m64 -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-6.5/bin/nvcc -I"/media/mehran/Cloud/University Stuff/UCF/CudaHoG" -G -g -pg -O0 -Xcompiler -fPIC --compile --target-cpu-architecture x86 -m64  -x c++ -o  "$@" -c "$<" -std=c++11
	@echo 'Finished building: $<'
	@echo ' '


