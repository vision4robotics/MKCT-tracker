% GMSD - measure the image quality of distorted image 'Y2' with the reference image 'Y1'.
% 
% inputs:
% 
% Y1 - the reference image (grayscale image, double type, 0~255)
% Y2 - the distorted image (grayscale image, double type, 0~255)
% 
% outputs:

% score: distortion degree of the distorted image
% quality_map: local quality map of the distorted image

% This is an implementation of the following algorithm:
% Wufeng Xue, Lei Zhang, Xuanqin Mou, and Alan C. Bovik, 
% "Gradient Magnitude Similarity Deviation: A Highly Efficient Perceptual Image Quality Index",
% http://www.comp.polyu.edu.hk/~cslzhang/IQA/GMSD/GMSD.htm

% origin format: score = GMSD_factor(Y1, Y2)
% but only used for one channel
% modified for RGB channel:
function score = GMSD_factor(ref, comp)
% score = [GMSD(Y1(:,:,1), Y2(:,:,1)) + ...
%          GMSD(Y1(:,:,2), Y2(:,:,2)) + ...
%          GMSD(Y1(:,:,3), Y2(:,:,3))] /3;
score = mean([GMSD(ref(:,:,1), comp(:,:,1)), ...
              GMSD(ref(:,:,2), comp(:,:,2)), ...
              GMSD(ref(:,:,3), comp(:,:,3))]);
end

function score_gray = GMSD(Y1, Y2)
T = 170; 
Down_step = 2;
dx = [1 0 -1; 1 0 -1; 1 0 -1]/3;
dy = dx';

aveKernel = fspecial('average',2);
aveY1 = conv2(Y1, aveKernel,'same');
aveY2 = conv2(Y2, aveKernel,'same');
Y1 = aveY1(1:Down_step:end,1:Down_step:end);
Y2 = aveY2(1:Down_step:end,1:Down_step:end);

IxY1 = conv2(Y1, dx, 'same');     
IyY1 = conv2(Y1, dy, 'same');    
gradientMap1 = sqrt(IxY1.^2 + IyY1.^2);

IxY2 = conv2(Y2, dx, 'same');     
IyY2 = conv2(Y2, dy, 'same');

gradientMap2 = sqrt(IxY2.^2 + IyY2.^2);
quality_map = (2*gradientMap1.*gradientMap2 + T) ./(gradientMap1.^2+gradientMap2.^2 + T);
score_gray = std2(quality_map);
end
