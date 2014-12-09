clear all
filterSize = 7;
image = im2single(imread('images/1.png'));
% arra = gpuArray(image);

[height width numChannels] = size(image);

tic
cudaResult = CudaBoxFilter(image, filterSize);
a = toc;
b = a * 1000;
fprintf('CUDA   was %.2f milliseconds (%.4f seconds) \n', b, a);
c = a;
figure, imshow(cudaResult,[]);

tic
mask = (1/(filterSize * filterSize)) * ones(filterSize, filterSize);
matFil = imfilter(image, mask, 'same');
a = toc;
b = a * 1000;
fprintf('MATLAB was %.2f milliseconds (%.4f seconds) \n', b,a );
m = a;

speedup = m / c;
fprintf('\nSpeed up was %f\n\n', speedup);
figure, imshow(matFil);

% YAY! Test passed :-) I rule ;-]