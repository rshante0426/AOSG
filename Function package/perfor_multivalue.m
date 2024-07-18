function [falsePositives, falseNegatives, overallError, pixelCorrelationCoefficient, kappa, multiValueImage, F1Score] = perfor_multivalue(im, imRef)
% Performance evaluation function for multi-value images
% Measures: true positives (tp), false positives (fp), true negatives (tn), false negatives (fn)
% Key formulae: tp + fn = Nc (number of changed pixels), tn + fp = Nu (number of unchanged pixels)

% Convert images to double precision for accurate computations
im = double(im(:, :, 1));
imRef = double(imRef);
imRef(imRef == 0) = 2;  % Mark unchanged pixels in reference with 2
imRef(imRef == 255) = 254;  % Mark changed pixels in reference with 254

[A, B] = size(im);  % Image dimensions
N = A * B;  % Total number of pixels
multiValueImage = zeros(A, B);  % Initialize multi-value observation map

% Determine the number of unchanged pixels in the reference image
unchangedCount = length(find(imRef == 2));
% Determine the number of changed pixels in the reference image
changedCount = length(find(imRef == 254));

% Compute the difference between the test and reference images
imgDiff = im - imRef;

for i = 1:A
    for j = 1:B
        % Classify pixels based on differences
        if imgDiff(i,j) == -2   % Correctly identified unchanged
            multiValueImage(i,j) = 0;
        elseif imgDiff(i,j) == 1   % Correctly identified changed
            multiValueImage(i,j) = 255;
        elseif imgDiff(i,j) == -254   % False negative (missed change)
            multiValueImage(i,j) = 180;
        elseif imgDiff(i,j) == 253   % False positive (incorrect change detected)
            multiValueImage(i,j) = 100;
        end
    end
end

% Count occurrences of each category in the multi-value image
falsePositives = length(find(multiValueImage == 100));
falseNegatives = length(find(multiValueImage == 180));
trueNegatives = length(find(multiValueImage == 0));
truePositives = length(find(multiValueImage == 255));

overallError = falsePositives + falseNegatives;
pixelCorrelationCoefficient = (truePositives + trueNegatives) / N;

% Kappa coefficient calculation
predictedAgreement = ((truePositives + falsePositives) * (truePositives + falseNegatives) + ...
       (falseNegatives + trueNegatives) * (falsePositives + trueNegatives)) / N^2;
kappa = (pixelCorrelationCoefficient - predictedAgreement) / (1 - predictedAgreement);

% F1 Score calculation
precision = truePositives / (truePositives + falsePositives);
recall = truePositives / (truePositives + falseNegatives);
F1Score = 2 * precision * recall / (precision + recall);

end
