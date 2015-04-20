
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#pragma warning(disable:4819)
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

#include <fstream>
#include <iostream>
#include <iomanip>
#include "codefull/ImageUtils.h"
#include "codefull/FloatImage.h"
#include "codefull/PreciseTimer.h"

using namespace codefull;
using namespace std;

int main(int argc, char *argv[]) {

	FloatImage image("images/Lena.pgm");

	PreciseTimer timer;

	timer.start();
	FloatImage convolved = image.convolve(ImageUtils::getBoxKernel(7));
	convolved.display();
	timer.stopAndLog("Box filter");
	convolved.save("Lena-filtered.png");

	timer.start();
	CudaMemory<float> result = ImageUtils::extractHOGFeatures(image);
	timer.stopAndLog("Feature extraction");

	ofstream out("result.m");
	out << "cudaHogC = [";

	for (int i = 0 ; i < result.getWidth() ; i++) {
		out << result.getHostData()[i] << (i != result.getWidth() - 1 ? ", " : "");
	}
	out << "];";
	out.close();
	exit(0);
}
