function [dif_fw, dif_bw] = CalculateChangeMagnitude(X, Y, opt)
% Calculates the magnitude of change between two sets of data (X and Y) 
% using iterative KNN searches and K-means clustering for segmentation.

% Initialization
N = size(X, 1); % Total number of samples
Kmax = round(opt.klamda * opt.Ns);
Kmin = 0.1 * round(opt.klamda * opt.Ns);
% Initialize arrays to store difference values across iterations
dif_x1 = zeros(N,opt.Niter);
dif_y1 = zeros(N,opt.Niter);
dif_x2 = zeros(N,opt.Niter);
dif_y2 = zeros(N,opt.Niter);
F_x = zeros(N,opt.Niter);
F_y = zeros(N,opt.Niter);
dif_x = zeros(N,opt.Niter);
dif_y = zeros(N,opt.Niter);

iter = 1;
X_library = X;
Y_library = Y;

% Begin iterative process
while iter <= opt.Niter
    % Perform KNN search for X and Y
    [idx, distX] = knnsearch(X_library,X,'k',Kmax);
    [idy, distY] = knnsearch(Y_library,Y,'k',Kmax);
    
    % Dynamic K selection
    k_scale_len_X = K_select(distX,N,Kmax,opt.nc);
    k_scale_len_Y = K_select(distY,N,Kmax,opt.nc);

    for i = 1:N
        % Calculate means of neighbor blocks
        X_idx_mean = mean(X_library(idx(i,1:k_scale_len_X(i)),:), 1);
        X_idy_mean = mean(X_library(idy(i,1:k_scale_len_X(i)),:), 1);
        Y_idy_mean = mean(Y_library(idy(i,1:k_scale_len_Y(i)),:), 1);
        Y_idx_mean = mean(Y_library(idx(i,1:k_scale_len_Y(i)),:), 1);
        
        % Compute various distances
        di_x = pdist2(X_idx_mean, X(i,:)).^2;
        di_y = pdist2(Y_idy_mean, Y(i,:)).^2;
        di_x_y1 = pdist2(X_idy_mean, X(i,:)).^2;
        di_y_x1 = pdist2(Y_idx_mean, Y(i,:)).^2;
        dif_x2(i) = pdist2(X_idy_mean, X_idx_mean).^2;
        dif_y2(i) = pdist2(Y_idy_mean, Y_idx_mean).^2;
        
        % Combine differences considering forward and backward mapping
        dif_x(i, iter) = abs(di_x_y1 + dif_x2(i) - di_x);
        dif_y(i, iter) = abs(di_y_x1 + dif_y2(i) - di_y);
    end
    
    % Outlier removal
    dif_x(:, iter) = remove_outlier(dif_x(:, iter));
    dif_y(:, iter) = remove_outlier(dif_y(:, iter));
    
    % Initial binary segmentation using K-means
    [~,center_x] = kmeans(dif_x(:, iter),2);
    eta_x = opt.alpha * (min(center_x) + max(center_x));
    [~,center_y] = kmeans(dif_y(:, iter),2);
    eta_y = opt.alpha * (min(center_y) + max(center_y));
    
    % Update libraries and Kmax
    id_unchange_x = find(dif_x(:, iter) - eta_x <= 0);
    X_library = X(id_unchange_x, :);
    Y_library = Y(id_unchange_x, :);
    Nuc = length(id_unchange_x);
    Kmax = round(opt.klamda * Nuc);
    
    % Compute objective functions
    for i = 1:N
        dif_x_delt_c = abs(abs(dif_x(i, iter) - mean(dif_x(idx(i,1:k_scale_len_X(i)), iter))) - max(dif_x(:, iter)));
        dif_x_delt_uc = abs(abs(dif_x(i, iter) - mean(dif_x(idx(i,1:k_scale_len_X(i)), iter))) - min(dif_x(:, iter)));
        dif_x_pc = (dif_x(i, iter) - mean(dif_x(idx(i,1:k_scale_len_X(i)), iter)) > eta_x);
        dif_x_pu = ~dif_x_pc;
        F_x(i, iter) = dif_x_pc .* dif_x_delt_c + dif_x_pu .* dif_x_delt_uc;
        
        dif_y_delt_c = abs(abs(dif_y(i, iter) - mean(dif_y(idy(i,1:k_scale_len_Y(i)), iter))) - max(dif_y(:, iter)));
        dif_y_delt_uc = abs(abs(dif_y(i, iter) - mean(dif_y(idy(i,1:k_scale_len_Y(i)), iter))) - min(dif_y(:, iter)));
        dif_y_pc = (abs(dif_y(i, iter) - mean(dif_y(idy(i,1:k_scale_len_Y(i)), iter))) > eta_y);
        dif_y_pu = ~dif_y_pc;
        F_y(i, iter) = dif_y_pc .* dif_y_delt_c + dif_y_pu .* dif_y_delt_uc;
    end
    
    % Check convergence criteria
    if iter > 1
        F_x_eps = sum(abs(F_x(:, iter-1) - F_x(:, iter))) / N;
        F_y_eps = sum(abs(F_y(:, iter-1) - F_y(:, iter))) / N;
        if F_x_eps < opt.epsilon && F_y_eps < opt.epsilon
            break;
        end
    end
    iter = iter+1;
end

if iter > opt.Niter
    dif_fw = dif_x(:, iter - 1);
    dif_bw = dif_y(:, iter - 1);
else
    dif_fw = dif_x(:, iter);
    dif_bw = dif_y(:, iter);
end

end