function pat_idx = patchfea_AOSG(image, x, ps)
% Computes patch features including mean, variance, and gradient sum for image patches.
%
% Inputs:
% image - The original image from which patches are derived.
% x - Data matrix with m rows (number of patches) and n columns (feature vector of each patch).
% ps - Patch size used for feature extraction.
%
% Output:
% pat_idx - Concatenated matrix of normalized mean, variance, and gradient sums for each patch.

% Compute mean values across columns
miu = mean(x, 2);

% Compute variance across rows
sigma = var(x, 1, 2);

% Gradient calculation parallel loop
[~, ~, B] = size(image);
n = size(x,1);
dx = [1 0 -1; 1 0 -1; 1 0 -1]/3; % Horizontal gradient kernel
dy = dx'; % Vertical gradient kernel (transposed dx)

% Initialize gradient sum array
gradient_sum = zeros(n,1);

parfor i = 1:n
    % Reshape the feature vector back to the original patch size
    patch = reshape(x(i,:), 2*ps+1, 2*ps+1, B);
    
    % Convert to grayscale if the patch is RGB
    if B > 1
        patch = rgb2gray(patch);
    end
    
    % Compute gradients
    patch_Ix = conv2(double(patch), dx, 'same');
    patch_Iy = conv2(double(patch), dy, 'same');
    gradient = sqrt(patch_Ix.^2 + patch_Iy.^2);
    
    % Sum gradients for the patch
    gradient_sum(i) = sum(gradient(:));
end

% Normalize features to a common scale
miu_min = min(miu);
miu_max = max(miu);
sigma = rescale(sigma, miu_min, miu_max); % Normalize variance
gradient_sum = rescale(gradient_sum, miu_min, miu_max); % Normalize gradient sum

% Concatenate normalized features into pat_idx
pat_idx = [miu, sigma, gradient_sum];
end

% Helper function to rescale values to a specified range
function scaled_values = rescale(values, min_val, max_val)
    scaled_values = (values - min(values)) ./ (max(values) - min(values)) * (max_val - min_val) + min_val;
end