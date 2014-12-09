clear all
image = im2single(imread('images/2.png'));
[height width numChannels] = size(image);

fprintf('\nImage size is %d x %d\n', width, height);

tic
cudaHog = CudaHoG(image);
a = toc;
b = a * 1000;
fprintf('CUDA   was %.2f milliseconds (%.4f seconds) \n', b, a);
c = a;

tic
[matHog vis] = extractHOGFeatures(image);
a = toc;
b = a * 1000;
fprintf('MATLAB was %.2f milliseconds (%.4f seconds) \n', b,a );
m = a;

speedup = m / c;
fprintf('\nSpeed up was %f\n\n', speedup);

figure, imshow(image), figure, imshow(HOGpicture(reshapeHoG(cudaHog, width, height), 16));

tic
vlHog = vl_hog(image,8, 'variant', 'dalaltriggs');
a = toc;

vlimhog = vl_hog('render', vlHog, 'variant', 'dalaltriggs');
% figure, imshow(vlimhog);

b = a * 1000;
fprintf('VLFEAT was %.2f milliseconds (%.4f seconds) \n', b,a );
m = a;