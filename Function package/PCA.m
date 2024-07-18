function [m, W] = PCA(patterns, dimension)

%Reshape the data points using the principal component analysis
%Inputs:
%	patterns        - Input patterns
%	dimension		- Number of dimensions for the output data points
%
%Outputs
%	patterns		- New patterns
%	UW				- Reshape martix
%   m               - Original pattern averages
%   W               - Eigenvector matrix 

[r,c] = size(patterns);

if (r < dimension),
   disp('Required dimension is larger than the data dimension.')
   disp(['Will use dimension ' num2str(r)])
   dimension = r;
end

%Calculate cov matrix and the PCA matrixes
m           = mean(patterns')';
S			= ((patterns - m*ones(1,c)) * (patterns - m*ones(1,c))');
[V, D]	    = eig(S);
W           = V(:,1:dimension)';
%W			= V(:,r-dimension+1:r)';
%U			= S*W'*inv(W*S*W');
%Calculate new patterns
%patterns    = W*patterns;