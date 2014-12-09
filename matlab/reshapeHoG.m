function [ reshaped ] = reshapeHoG( hogFeatures, width, height )
%RESHAPEHOG Summary of this function goes here
%   Detailed explanation goes here

newWidth = floor((width / 8)) - 1;
newHeight = floor((height / 8)) - 1;
reshaped = zeros(newHeight, newWidth, 36);
offset = 0;

for j=1:newHeight
    for i=1:newWidth
        reshaped(j,i,:) = hogFeatures(offset + 1 : offset + 36);
        offset = offset + 36;
    end
end

% reshaped = reshape(hogFeatures, newHeight, newWidth, 36);



end