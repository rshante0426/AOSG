function y = tonorm(image)
% Data normalization
image = double(image);
[~, ~, d] = size(image);
for i = 1:d
    image(:,:,i) = (image(:,:,i) - min(min(image(:,:,i)))) ./ (max(max(image(:,:,i)))- min(min(image(:,:,i))));
end
y = image;
end