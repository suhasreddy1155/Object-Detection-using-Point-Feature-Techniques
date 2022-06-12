clc
clear all
%Reading the Input Image
a=imread('eleph.png');

%converting the images to grayscale
b=rgb2gray(a);
c=rgb2gray(imread('elephnoise.png'));

%Detect orb Features from both the Input and Altered Image
tic
opoints=detectORBFeatures(b);
otpoints=detectORBFeatures(c);
toc
%Plotting the orb features of Scene Image
figure;
imshow(b);
title('Strongest Feature Points from Input Image');
hold on;
%Selecting strong features among all features from Input image
plot(selectStrongest(opoints,size(opoints,1)));

%Plotting the orb features of Target Image
figure;
imshow(c);
title('Strongest Feature Points from Altered Image');
hold on;
%Selecting strong features among all features from Altered Image
plot(selectStrongest(otpoints,size(otpoints,1)));

%Extracting the Descriptors from both images
[otFeatures, otpoints] = extractFeatures(c, otpoints);
[oFeatures, opoints] = extractFeatures(b, opoints);

%Matching the features using their descriptors.
tPairs = matchFeatures(otFeatures, oFeatures);

%Displaying putatively matched features.
matchedAltPoints = otpoints(tPairs(:, 1), :);
matchedInpPoints = opoints(tPairs(:, 2), :);
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
fprintf('Image 1: %d\nImage 2: %d\nAccuracy : %f\n',size(opoints,1),size(otpoints,1),Accuracy);