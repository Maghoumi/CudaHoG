
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#pragma warning(disable:4819)
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

#include <fstream>
#include <iostream>
#include <iomanip>

#include "codefull/ImageUtils.cuh"
#include "codefull/FloatImage.h"
#include "codefull/PreciseTimer.h"

using namespace codefull;
using namespace std;

int main(int argc, char *argv[]) {
	FloatImage image("Lena.pgm");

	PreciseTimer timer;

	timer.start();
	image.convolveInPlace(ImageUtils::getBoxKernel(5), NppiSize{5,5});
	timer.stopAndLog("Box filter");
	image.save("filtered.jpg");

	timer.start();
	CudaMemory<float> result = ImageUtils::extractHOGFeatures(image);
	timer.stopAndLog("Feature extraction");

	ofstream out("result.m");
	out << "cudaHogC = [";

	for (int i = 0 ; i < result.getWidth() ; i++) {
		out << result.getHostData()[i] << (i != result.getWidth() - 1 ? ", " : "");
	}
	out << "]";
	out.close();
	exit(Success);
}
