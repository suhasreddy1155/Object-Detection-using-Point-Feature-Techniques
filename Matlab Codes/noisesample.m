clc
clear all
%Reading the Reference Scene
a=imread('eleph.png');

%converting the images to grayscale
b=rgb2gray(a);
c=imnoise(b,'salt & pepper',0.2);
imshow(c);