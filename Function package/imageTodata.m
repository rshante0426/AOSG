function [PGM] = imageTodata(image, patchSize, stepSize)
% Converts an image into a data matrix by extracting patches with a given size
% and step size. The function iterates over the image, extracting square patches
% defined by 'patchSize', stepping through the image with 'stepSize',
% and stacking these patches into a column vector within the output matrix 'PGM'.

% Obtain dimensions of the input image including color bands (assuming RGB)
[H, W, B] = size(image);

% Generate sequences for x and y indices based on step size and patch size
i_x = (1+patchSize):stepSize:(H-patchSize);
% Ensure the last index reaches up to (H-patchSize)
if i_x(end) ~= (H-patchSize)
    i_x = [i_x, H-patchSize];
end

j_y = (1+patchSize):stepSize:(W-patchSize);
% Ensure the last index reaches up to (W-patchSize)
if j_y(end) ~= (W-patchSize)
    j_y = [j_y, W-patchSize];
end

% Initialize the output matrix PGM and counter k
k = 1;
% Preallocate PGM for efficiency
PGM = zeros((patchSize*2+1)^2 * B, length(i_x)*length(j_y));

% Iterate over the defined grid to extract patches
for i = 1:length(i_x)
    for j = 1:length(j_y)
        % Calculate bounds for the current patch
        i_x_d = i_x(i) - patchSize;
        i_x_u = i_x(i) + patchSize;
        j_y_d = j_y(j) - patchSize;
        j_y_u = j_y(j) + patchSize;
        
        % Extract the patch from the image including all color bands
        dataPatch = image(i_x_d:i_x_u, j_y_d:j_y_u, 1:B);
        
        % Flatten the patch and place it as a column in PGM
        PGM(:, k) = dataPatch(:);
        % Increment the column index
        k = k + 1;
    end
end
end