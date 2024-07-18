function k_scale = K_select(distX, N, k, nc)
% This function determines the optimal number of neighborhood patches at varying scales
% for each sample patch based on clustering distances.
%
% Inputs:
% - distX: Distance matrix representing the distances between a patch and other patches.
% - N: Total number of sample patches.
% - k: Maximum number of similar patches considered.
% - nc: Number of clusters for K-means clustering.
%
% Output:
% - k_scale: Optimal number of neighborhood patches per sample patch.

% Initialize matrices to hold K-means results and labels for each patch
distX_km = ones(N, k);
distX_km_label = ones(N, nc);

% Parallel loop to perform K-means clustering for each block's distances
parfor i = 1:N
    % Extract distances for the i-th block
    distX_i = distX(i, :);
    
    % Apply K-means clustering with a maximum of 'nc' clusters
    % and a maximum of 200 iterations to ensure stability
    [distX_i_km, ~] = kmeans(distX_i(:), nc, 'MaxIter', 200);
    
    % Transpose to ensure compatibility if necessary (though in this context, it might be redundant)
    distX_i_km = distX_i_km.';
    
    % Obtain unique labels from the clustering result
    distX_km_label(i, :) = unique(distX_i_km, 'stable');
    
    % Store the clustering results for each block
    distX_km(i, :) = distX_i_km;
end

% Initialize vector to store the optimal number of neighborhood blocks per sample block
k_scale = zeros(N, 1);

% Iterate over each sample block to determine the optimal scale (cluster with the most representatives)
for i = 1:N
    % Find the cluster label occurring first in the sorted order (arbitrary choice for 'most representative')
    primary_cluster = distX_km_label(i, 1);
    
    % Count the occurrences of this label in the block's clustering results
    k_scale(i) = sum(distX_km(i, :) == primary_cluster);
end

end
