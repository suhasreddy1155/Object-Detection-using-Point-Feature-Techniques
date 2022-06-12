clc
clear all
%Reading the Reference Scene
a=imread('eleph.png');

%converting the images to grayscale
b=rgb2gray(a);
c=2*b;

%Detect Surf Features from both the Reference and target Image
tic
bpoints=detectBRISKFeatures(b);
btpoints=detectBRISKFeatures(c);
toc
%Plotting the surf features of Scene Image
figure;
imshow(b);
title('Strongest Feature Points from Input Image');
hold on;
%Selecting 300 strong features among all features from scene image
plot(selectStrongest(bpoints,size(bpoints,1)));

%Plotting the surf features of Target Image
figure;
imshow(c);
title('Strongest Feature Points from Altered Image');
hold on;
%Selecting 100 strong features among all features from Target Image
plot(selectStrongest(btpoints,size(btpoints,1)));

%Extracting the Descriptors from both images
[btFeatures, btpoints] = extractFeatures(c, btpoints);
[bFeatures, bpoints] = extractFeatures(b, bpoints);

%Matching the features using their descriptors.
tPairs = matchFeatures(btFeatures, bFeatures);

%Displaying putatively matched features.
matchedAltPoints = btpoints(tPairs(:, 1), :);
matchedInpPoints = bpoints(tPairs(:, 2), :);
figure;

%Display multiple image frames as rectangular montage
showMatchedFeatures(c , b, matchedAltPoints, ...
    matchedInpPoints, 'montage');
title('Matched Points (Including Outliers)');
[tform, inlierIdx] = ...
    estimateGeometricTransform2D(matchedAltPoints, matchedInpPoints,'affine');
inliertpoints   = matchedAltPoints(inlierIdx, :);
inlierpoints = matchedInpPoints(inlierIdx, :);
Accuracy=100*size(inlierpoints)/size(matchedAltPoints);
fprintf('Image 1: %d\nImage 2: %d\nAccuracy : %f\n',size(bpoints,1),size(btpoints,1),Accuracy);