function [img_color] = Gray2Color(img, row_num, uni_value, color)
% Convert grayscale label map to a color image
m = cell(row_num, 1);
n = cell(row_num, 1);

% Find indices of each unique value in the grayscale image
for i = 1:row_num
    [m{i}, n{i}] = find(img == uni_value(i));
end

img_color = repmat(img, 1, 1, 3);

% Initialize color image with grayscale image replicated across three channels
% Perform color replacement
for k=1:row_num
    for i=1:size(m{k})
        img_color(m{k}(i), n{k}(i), 1) = color{k}(1);
        img_color(m{k}(i), n{k}(i), 2) = color{k}(2);
        img_color(m{k}(i), n{k}(i), 3) = color{k}(3);
    end
end

% Ensure the image is in the correct data type for image display
img_color = uint8(img_color);

end