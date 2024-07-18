function [DI] = dataToimage(data, ps, ss, image)
% Converts data into an image representation based on defined patch sizes and steps.
%
% INPUTS:
% - data: Vector containing pixel values to be placed in the image.
% - ps: Patch size, defining the side length of the square area around each pixel.
% - ss: Step size, determining the distance between centers of consecutive patches.
% - image: Original image used for dimension reference.

% Get image dimensions, ignoring color channels
[H, W, ~] = size(image);
image_re = zeros(H,W); % Initialize a matrix to hold rearranged data

% Generate sequences for x and y indices based on step and patch size
i_x = (1+ps):ss:(H-ps);
if i_x(end) ~= (H-ps)
    i_x = [i_x, H-ps]; % Ensure the last index hits the bottom boundary
end
j_y = (1+ps):ss:(W-ps);
if j_y(end) ~= (W-ps)
    j_y = [j_y, W-ps]; % Ensure the last index hits the right boundary
end

% Populate the rearranged image with data elements
k = 1;
for i = 1:length(i_x)
    for j = 1:length(j_y)
        image_re(i_x(i),j_y(j)) = data(k);
        k = k+1;
    end
end

% Compute the final difference image (DI) based on local patch averages
for i = 1:H
    for j = 1:W
        % Determine local patch indices for averaging
        local_i = local_index(i, ps, ss, H);
        local_j = local_index(j, ps, ss, W);
        
        % Compute mean of the local patch in the rearranged image
        DI(i,j) = mean(mean(image_re(local_i, local_j)));
    end
end

% Local index calculation helper function embedded within dataToimage
local_x = local_index(i, ps, ss, H);
    function local_indices = local_index(index, ps, ss, totalDim)
        % Determines the local patch indices around a given index
        if index < 1+ss
            local_indices = 1+ps;
        elseif index >= floor((totalDim-2*ps-1)/ss)*ss+1+2*ps
            local_indices = totalDim-ps;
        else
            local_indices = (1+ps+max(0,ceil((index-2*ps-1)/ss))*ss):ss:min((1+ps+floor((index-1)/ss)*ss),totalDim);
        end
    end
end