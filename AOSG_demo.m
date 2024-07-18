%% Title: Unsupervised multimodal change detection based on adaptive optimization of structured graph
%% Abstract: This paper proposes a novel method for  multimodal change detection (MCD) using an 
% adaptive optimization of the structured graph (AOSG) to mine comparable structural features across multimodal images.
%% Journal: International Journal of Applied Earth Observation and Geoinformation
%% https://doi.org/10.1016/j.jag.2023.103630
%% If you find our work beneficial, please feel free to cite our paper as follows.
%% If the test results are not good, you can consider using only DI_fw or DI_bw.

%%
tic
clc; clear; close all;
%% Load Images --------------------------------------------------------------
% Define types of the two images
imageType_t1 = 'Opt'; % 'SAR' or 'Opt'
imageType_t2 = 'Opt'; % 'SAR' or 'Opt'

% Specify image file paths
basePath = 'F:\EXP_images\HeterogenerousData\Opt-Opt\image02';
file_image_t1 = 'Italy_1.bmp';
file_image_t2 = 'Italy_2.bmp';
file_Ref_gt = 'Italy_gt.bmp';

% Read images
image_t1 = imread(fullfile(basePath, file_image_t1));
image_t2 = imread(fullfile(basePath, file_image_t2));
Ref_gt = imread(fullfile(basePath, file_Ref_gt));

% Apply log transformation based on image type
if strcmpi(imageType_t1, 'SAR')
    fprintf('Image T1 is SAR type, applying log transformation...\n');
    image_t1 = logtrans(image_t1);
end

if strcmpi(imageType_t2, 'SAR')
    fprintf('Image T2 is SAR type, applying log transformation...\n');
    image_t2 = logtrans(image_t2);
end

image_t1 = tonorm(image_t1);
image_t2 = tonorm(image_t2);
figure, imshow(image_t1, []);axis on
figure, imshow(image_t2, []);axis on
figure,imshow(Ref_gt,[]),title('Change Reference Map');

[H, W, B] = size(image_t1);

%% Parameter setting--------------------------------------------------------------
opt.Ns = 10000; % Default patch number
opt.p_s = floor(sqrt([H * W] ./ opt.Ns)) ;
opt.delt_p = opt.p_s; 
opt.nc = 3;
opt.Niter = 6;
opt.epsilon = 0.005;
opt.klamda = 0.03;
opt.alpha = 0.5;

%% Data preprocessing
fprintf(['\n AOSG is running...... ' '\n'])
X = imageTodata(image_t1, opt.p_s, opt.delt_p);
Y = imageTodata(image_t2, opt.p_s, opt.delt_p);
X = X';
Y = Y';

% Calculate patch features.
X = patchfea_AOSG(image_t1, X, opt.p_s);
Y = patchfea_AOSG(image_t2, Y, opt.p_s);

%% Calculated change intensity of each patch ----------------------------------
[dif_fw, dif_bw] = CalculateChangeMagnitude(X, Y, opt);

% Show the CIMs
DI_fw = dataToimage(dif_fw, opt.p_s, opt.delt_p, image_t1);
DI_fw = tonorm(DI_fw);
figure,imshow(DI_fw,[]);
colormap(parula);
colorbar;
DI_bw = dataToimage(dif_bw, opt.p_s, opt.delt_p, image_t1);
DI_bw = tonorm(DI_bw);
figure,imshow(DI_bw,[]);
colormap(parula);
colorbar;
[DI_fusion,C,D] = BayesFusion(DI_fw,DI_bw);
DI_fusion = remove_outlier(DI_fusion);
DI_fusion = tonorm(DI_fusion);
figure,imshow(DI_fusion,[]);
colormap(parula);
colorbar;

%% Change Detection Map Generation ----------------------------------------------
CM = CM_Generation(DI_fusion, opt, H, W);
figure,imshow(CM,[]);title('Change Map');

%% Accuracy Assessment
n=500;
[TPR_fusion, FPR_fusion]= Roc_plot(DI_fusion,Ref_gt,n);
[AUC_fusion, Ddist_fusion] = AUC_Diagdistance(TPR_fusion, FPR_fusion);

[fp,fn,oe,oa,kappa,multivalue,F1]=perfor_multivalue(CM, Ref_gt);
fprintf('Accuracy OA is: %4.4f Kappa coefficient KC is: %4.4f F1 score is: %4.4f \n', oa, kappa, F1);
uni_value = [0;100;180;255];
row_num = size(uni_value, 1);
color = cell(row_num, 1);
color{1} = [255, 250, 236];  
color{2} = [255, 78, 0];
color{3} = [28, 28, 28];
color{4} = [29, 121, 192];
[img_color] = Gray2Color(multivalue, row_num, uni_value, color);
figure, imshow(img_color);

toc