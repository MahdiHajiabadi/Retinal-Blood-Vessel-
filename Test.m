clear all;
 clc
 close all
 load '12.mat'
 im = imread('C:\MATLAB\retina\dataset drive\DRIVE\test\images/01_test.tif');
Truth = imread('C:\MATLAB\retina\dataset drive\DRIVE\test\1st_manual/01_manual1.gif');
%mask = imread('E:\MATLAB\DRIVE\test\mask/01_test_mask.gif');
% mask = mask(30:564, 15:550);
Truth = Truth(30:564, 15:550);
% mask = double(mask)/255;
Truth = double(Truth)/255;
ImCrop = im(30:564, 15:550,1:3);
 %figure;
 %imshow(ImCrop);
% % % =========================== Trnasformation Phase 
%%=============================1: Lab transformation 
 colorTransform = makecform('srgb2lab');
lab = applycform(ImCrop,colorTransform);
 %figure;
 %imshow(lab);

%%%=============================2,3: YCbCr trnasformation and Guassian Trnasformation
transformation = [0.257, 0.504, 0.098; -0.148, -0.291 , 0.439 ; 0.439 -0.368 -0.071];
Gaussian = [0.06, 0.63, 0.27;0.3 , 0.04 , -0.35; 0.34 , -0.6 0.17];
YCbCr = ImCrop;
GImage = ImCrop;
[Ix Iy Iz] = size(ImCrop);
Temp = zeros(3,1);
for i = 1:Ix
    for j = 1:Iy
        Temp(1) = ImCrop(i,j,1); 
        Temp(2) = ImCrop(i,j,2);
        Temp(3) = ImCrop(i,j,3);
        YCbCr(i,j,:) = transformation * Temp + [16;128;128];
        GImage(i,j,:) = Gaussian * Temp;
    end 
end
 %figure;
 %imshow(YCbCr(i,j,:));
 %figure;
 %imshow(GImage(i,j,:));
%%%% ================================= Final Trnasformation Image
G=ImCrop(:,:,2);
F = YCbCr(:,:,1);
D = lab(:,:,1);
GO = GImage(:,:,1);
%figure;
%imshow(G);
%figure;
%imshow(D);
%figure;
%imshow(F);
%figure;
%imshow(GO);
%%%=================== CLAHE (Contrast-limited Adaptive Histogram Equalization) algorithm for contrast Enhancement
Green =  adapthisteq(G,'clipLimit',0.01,'Distribution','uniform');
Y =  adapthisteq(F,'clipLimit',0.01,'Distribution','uniform');
L =  adapthisteq(D,'clipLimit',0.01,'Distribution','uniform');
G1 =  adapthisteq(GO,'clipLimit',0.01,'Distribution','uniform');

%figure;
%imshow(Green);
%figure;
%imshow(Y);
%figure;
%imshow(L);
%figure;
%imshow(G1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Gabor Filtering on the 4 Channels (Green,Y,L,G1 )
Input = cell(4,1);
Input{1} = L;
Input{2} = G1;
Input{3} = Y;
Input{4} = Green;
theta   = 0;    %%%% theta -> orientation
lambda  = [9 10 11];   % %%%% lambda -> Wave Length
psi     = [-pi pi]; %%% psi ->phase oofset
gamma   = 0.5; %%%%% gamma ->aspect ratio
bw      = 1;   %%% bw -> bandwidth
N       = 24;  %%% N -> Number of orientation
% % % 
Feature = zeros(size(G1,1),size(G1,2),13);
for i =1:4
    for j = 1:3
    KHSH  = Input{i};
   GG_example = gabor_example(KHSH,lambda(j),theta,psi,gamma,bw,N);
   % figure
  %  imshow(GG_example);
        
    %str = num2str(counter);
    %path = strcat(str,'.mat');
    %save(path,'dd');
   maxIntensify = max(max(GG_example));
   perVal = maxIntensify * .1;
   GG_example(GG_example<=perVal)=0;
   GG_example(GG_example>perVal)=1;
  % figure
  %imshow(GG_example);
   Feature(:,:,(i-1)*3 + j) = GG_example;
   
 %   Gab = Feature(:,:,(i-1)*3 + j);
%    str = num2str(counter);
 %   path = strcat(str,'.mat');
  %  save('Gab');
%    imshow(GG_example)

    end
end
Green = double(Green)/255;
Feature(:,:,13) = Green;
%gre=Feature(:,:,13);
 % str = num2str(counter);
  %  path = strcat(str,'.mat');
  %  save('gre');
   % imshow(GG_example)
% % % ======================Unsupervised and supervised learning phase
% % % % % % % % % % % % % % % % / PCA Implementaion
[m, n, d] = size(Feature);
Dimension = m*n;
X = reshape(Feature, Dimension, d);
%   str  = strcat('features',num2str(counter),'.csv');
      % csvwrite(str,X);
[coeff,score] = princomp(X);
% % % % % % % % % % % % % % % % % % % % % K-means Algorithms
 Res = kmeans(score,2,'distance','cosine');

No_Cluster1 = length(Res==1);
No_Cluster2 = length(Res==2);
if (No_Cluster1>No_Cluster2)
    Pixel = 1;
else
    Pixel = 2;
end
Res(Res==Pixel) = 0;
Res(Res==(2 - Pixel + 1)) = 1;
IDX = find(Res==0);
XFinal = X(IDX,:);
YFinal = Y(IDX);
Label = tt.predict(XFinal);
tic
toc
Res(IDX) = Label;       
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %  Post  Processing
Temp = vec2mat(Res,size(G1,1));
FinalImage1  = transpose(Temp);
 
figure
 imshow(FinalImage1);
 
FinalImage = 1 - FinalImage1;

figure
 imshow(FinalImage);
Label1 = length(find(FinalImage==1));
Label0 = length(find(FinalImage==0));
if (Label1>Label0)
    FinalImage = 1 - double(FinalImage);
end
mask= createmask(FinalImage);
Final = (mask .*(FinalImage));
% Final = mask - FinalImage;

figure,imshow(Final)

se1 = strel('disk',1);
 
se2 = strel('disk',2);
Final1 = imdilate(Final,se1);
Final2 = imerode(Final1,se2);
%figure,imshow(Final2)
     
% BW2 = bwmorph(Final,'remove');
% figure
% imshow(BW2)
% BW3 = bwmorph(BW2,'majority');
%figure
%imshow(BW3);
% BW4 = bwmorph(BW3,'bridge');
% figure
% imshow(BW4);

% % % % % % % % % % /Evaluation Sections
 [m, n] = size(Final2);
 Dimension = m*n;
 Predicted = reshape(Final2, Dimension,1);
Real = reshape(Truth, Dimension,1);
TP = sum(Predicted==1 & Real==1);
TN = sum(Predicted==0 & Real==0);
FP = sum(Predicted==1 & Real==0);
FN = sum(Predicted==0 & Real==1);
Accuracy = (TP + TN)/(TP + TN + FP + FN);
Sensitivity = TP/(TP + FN);
Specificity = TN/(TN + FP);
PPV = TP/(TP + FP);
A = table(Accuracy,Sensitivity,Specificity,PPV)