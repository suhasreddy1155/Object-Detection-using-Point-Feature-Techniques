clc
clear all
%Reading the Reference Scene
a=imread('scene.png');

% Read the target image (Ex:Box Image)
c=imread('Box.png');

%converting the images to grayscale
b=rgb2gray(a);
d=rgb2gray(c);

%Detect Surf Features from both the Reference and target Image
disp('Extracting Features');
tic
points=detectBRISKFeatures(b);
tpoints=detectBRISKFeatures(d);
toc

%Plotting the surf features of Scene Image
figure;
imshow(b);
title('Strongest Feature Points from Image');
hold on;
%Selecting 300 strong features among all features from scene image
plot(selectStrongest(points, 300));

%Plotting the surf features of Target Image
figure;
imshow(c);
title('Strongest Feature Points from Test Image');
hold on;
%Selecting 100 strong features among all features from Target Image
plot(selectStrongest(tpoints, 100));

%Extracting the Descriptors from both images
[tFeatures, tpoints] = extractFeatures(d, tpoints);
[Features, points] = extractFeatures(b, points);

%Matching the features using their descriptors.
disp('Matching Features');
tic
tPairs = matchFeatures(tFeatures, Features);
toc

%Displaying putatively matched features.
matchedBoxPoints = tpoints(tPairs(:, 1), :);
matchedScenePoints = points(tPairs(:, 2), :);
figure;

%Display multiple image frames as rectangular montage
showMatchedFeatures(d, b, matchedBoxPoints, ...
    matchedScenePoints, 'montage');
title('Putatively Matched Points (Including Outliers)');

%Locating the object in the Reference Image
%estimateGeometricTransform2D calculates the transformation relating the matched points,
%while eliminating outliers. This transformation allows us to localize the object in the scene.
%affine - a linear mapping method that preserves points, straight lines, and planes
[tform, inlierIdx] = ...
    estimateGeometricTransform2D(matchedBoxPoints, matchedScenePoints,'affine');
inliertpoints   = matchedBoxPoints(inlierIdx, :);
inlierpoints = matchedScenePoints(inlierIdx, :);

%For Drawing the box around the target image
tPolygon = [1, 1;...                           % top-left
        size(d, 2), 1;...                 % top-right
        size(d, 2), size(d, 1);... % bottom-right
        1, size(d, 1);...                 % bottom-left
        1, 1];    

%applies the forward transformation of tform to the input coordinate
%matrix tPolygon and returns the coordinate matrix newtPolygon
newtPolygon = transformPointsForward(tform, tPolygon);

%Display the matching point pairs with the outliers removed
figure;
imshow(b);
hold on;
line(newtPolygon(:, 1), newtPolygon(:, 2), 'Color', 'R');
title('Detected Target Image');