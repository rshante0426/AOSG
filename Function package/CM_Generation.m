function CM = CM_Generation(DI_fusion, opt, H, W)
% Function to generate a binary change map from fused difference image
% using K-means clustering for thresholding and a patch-based approach to 
% eliminate isolated points.

% Thresholding using K-means clustering
% Perform k-means clustering on the flattened difference image
[~, center] = kmeans(DI_fusion(:), 2);
% Determine the two cluster centers (min and max)
center_1 = min(center);
center_2 = max(center);
% Compute the threshold as a weighted average of the cluster centers
thresh = opt.alpha * (center_1 + center_2);
% Apply thresholding to create the initial contrast map
CM = DI_fusion;
CM(DI_fusion > thresh) = 255; % Pixels above threshold set to white (255)
CM(DI_fusion <= thresh) = 0;  % Pixels below or equal threshold set to black (0)

% Elimination of isolated points using a patch system
patchSize = 2;
% Pad the contrast map to handle edge cases during patch analysis
CM_padded = padarray(CM,[patchSize,patchSize],'replicate');
for i = 1 + patchSize : H + patchSize
    for j = 1 + patchSize : W + patchSize
        % Extract the current window/patch
        window = CM_padded(i-patchSize:i+patchSize,j-patchSize:j+patchSize);
        % Count black (0) and white (255) pixels within the window
        if length(find(window==0))>length(find(window==255))
            CM_padded(i,j)=0;
        else
            CM_padded(i,j)=255;
        end
    end
end
% Remove padding to restore original image dimensions
CM=CM_padded(patchSize+1:H+patchSize,patchSize+1:W+patchSize);

% You might also choose Method PCA_Kmeans
% CM = PCA_Kmeans(DI_fusion,3,3);
end


       
