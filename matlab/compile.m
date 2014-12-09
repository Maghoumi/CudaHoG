% !unlink GPUHog.o
% !nvcc -O0 -std=c++11 -c GPUHog.cu -Xcompiler -fPIC -I/usr/local/MATLAB/R2014a/extern/include -I/usr/local/MATLAB/R2014a/toolbox/distcomp/gpu/extern/include -L./ -lCudaFreeImage;
% mex -g -largeArrayDims GPUHog.o -L/usr/local/cuda/lib64 -L/usr/local/MATLAB/R2014a/bin/glnxa64 -lcudart -lcufft -lmwgpu -L./ -lCudaFreeImage -lstdc++

!nvcc -O0 -std=c++11 -c CudaBoxFilter.cu -Xcompiler -fPIC -I/usr/local/MATLAB/R2014a/extern/include -I/usr/local/MATLAB/R2014a/toolbox/distcomp/gpu/extern/include -L./ -lCudaHoG;
mex -g -largeArrayDims CudaBoxFilter.o -L/usr/local/cuda/lib64 -L/usr/local/MATLAB/R2014a/bin/glnxa64 -lcudart -lcufft -lmwgpu -L./ -lCudaHoG -lstdc++
!unlink CudaBoxFilter.o

!nvcc -O0 -std=c++11 -c CudaHoG.cu -Xcompiler -fPIC -I/usr/local/MATLAB/R2014a/extern/include -I/usr/local/MATLAB/R2014a/toolbox/distcomp/gpu/extern/include -L./ -lCudaHoG;
mex -g -largeArrayDims CudaHoG.o -L/usr/local/cuda/lib64 -L/usr/local/MATLAB/R2014a/bin/glnxa64 -lcudart -lcufft -lmwgpu -L./ -lCudaHoG -lstdc++
!unlink CudaHoG.o