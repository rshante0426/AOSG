function logTransformedImage = logtrans(image)
% Perform log transformation on the input image
% image: Input image matrix
% logTransformedImage: Output image after log transformation and normalization

% Obtain dimensions of the input image
[rows, cols] = size(image);

% Convert the image to double precision for calculations
image = double(image);

% Add a small constant to avoid log(0) which is undefined
image = image + 0.01;

% Apply logarithmic transformation
logImage = log(image);

% Normalize the log-transformed image to range [0, 1]
logImageNormalized = (logImage(:) - min(logImage(:))) / (max(logImage(:)) - min(logImage(:)));

% Reshape the normalized vector back to the original image size
logTransformedImage = reshape(logImageNormalized, rows, cols);
end