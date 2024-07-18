function [CM] = PCA_Kmeans(DI,h,S)
%function [CM] = PCA_Kmeans(DI,h,s)
%h  : block size , h x h, h=3 default
%S  : dimension of feature vectors, S = 3 default;
%DI : 差异图
%CM : output change map where 0 (black) refers to change and 1 (white) refers to nochange
%Turgay Celik, April 2009, Implementation of "Unsupervised Change
%Detection in Satellite Images Using Principal Component Analysis and k-Means ClusteringTransform", 
%IEEE Geoscience and Remote Sensing Letters, 2009.

[H,W] = size(DI);
K_means = 2;
i_d = floor(h/2);
i_u = h - (i_d + 1);
j_d = floor(h/2);
j_u = h - (j_d + 1);
k = 1;
patterns = zeros(h*h,length(i_d+1:h:H-i_u)*length(j_d+1:h:W-j_u));
for i = i_d+1:h:H-i_u
    for j = j_d+1:h:W-j_u
        data = DI(i-i_d:i+i_u,j-j_d:j+j_u);
        patterns(:,k) = data(:);
        k = k + 1;
    end
end
[mean_vector, eigen_vectors] = PCA(patterns,S);
clear patterns;
imLRT = [repmat(DI(:,1),1,j_d) DI repmat(DI(:,W),1,j_u)];
imLR  = [repmat(imLRT(1,:),i_d,1) ; imLRT ; repmat(imLRT(size(imLRT,1),:),i_d,1)]; 
patterns_ = zeros(h*h,H*W);
k = 1;
for i = i_d+1:i_d+H
    for j = j_d+1:j_d+W
        data = imLR(i-i_d:i+i_u,j-j_d:j+j_u);
        patterns_(:,k) = (data(:) - mean_vector);
        %patterns_(:,k) = data(:);
        k = k + 1;
    end
end
%projection onto EigenVector space
%patterns_ = abs(eigen_vectors*patterns_);
patterns_ = (eigen_vectors*patterns_);
idx = kmeans(patterns_', K_means, 'EmptyAction', 'singleton','Maxiter',1000);
CM = zeros(H,W);
k = 1;
for i = 1:H
    for j = 1:W
        CM(i,j) = idx(k) - 1;
        k = k + 1;
    end
end
indx_1 = find(CM == 1);
indx_0 = find(CM == 0);
mean_1 = mean(DI(indx_1));
mean_0 = mean(DI(indx_0));
if(mean_0<mean_1)
    CM(indx_0) = 0;
    CM(indx_1) = 255;
else
    CM(indx_0) = 255;
    CM(indx_1) = 0;
end