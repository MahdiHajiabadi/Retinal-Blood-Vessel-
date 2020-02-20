clear all;
 clc
 TrainIDX =[];  TruthIDX = []; %MaskIDX = [];
 Acc = zeros(20,1);
 for i = 1:20
    TrainIDX =  [TrainIDX; strcat(num2str(i+20),'_training','.tif')];
    TruthIDX = [TruthIDX;strcat(num2str(i+20),'_manual1','.gif')];
    %MaskIDX = [MaskIDX; strcat(num2str(i+20),'_training_mask.gif')];
 end
 for counter = 1:20
    im = imread(TrainIDX(i,:));
%     mask = imread(MaskIDX(i,:));
    Truth = imread(TruthIDX(i,:));
% % % % % %     PreProcessing Function is called:
    Feature = Preprocessing(im,counter);
% % % % % % % % % % % % % 

%     mask = mask(30:564, 15:550);
    Truth = Truth(30:564, 15:550);
%     mask = double(mask)/255;
    Truth = double(Truth)/255;
    % % % % % % % % % % % % % % % % / PCA Implementaion
    [m, n, d] = size(Feature);
    Dimension = m*n;
    X = reshape(Feature, Dimension, d);
% % % % % %     The dataset will be saved X (matrix has 13 columns)
    str  = strcat('Input_features',num2str(counter),'.csv');
    Y = reshape(Truth,size(Truth,1) * size(Truth,2),1);
%     csvwrite(str,X);
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
    
    tt = ClassificationTree.fit(XFinal,YFinal);
    str = num2str(counter);
    path = strcat(str,'.mat');
    save(path,'tt');    
    Label = tt.predict(XFinal);
    Res(IDX) = Label;    
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %  Post  Processing
    
    Temp = vec2mat(Res,size(Truth,1));
    FinalImage  = transpose(Temp);
    
    %str  = strcat('F',num2str(counter),'.tif');
     %imwrite( FinalImage,str,'tif');
     
    FinalImage = 1 - FinalImage;
    Label1 = length(find(FinalImage==1));
    Label0 = length(find(FinalImage==0));
    if (Label1>Label0)
        FinalImage = 1 - double(FinalImage);
    end
     % str  = strcat('Final',num2str(counter),'.tif');
    % imwrite( FinalImage,str,'tif');
    
    [mask]= createmask(FinalImage);
     Final = (mask .*(FinalImage));
%     Final = mask - FinalImage;
    %str  = strcat('Gabor_',num2str(counter),'.tif');
     %imwrite(Final,str,'tif');
     
   se1 = strel('disk',1);
 
se2 = strel('disk',2);
Final1 = imdilate(Final,se1);
Final2 = imerode(Final1,se2);

%  str  = strcat('Final2disk',num2str(counter),'.tif');
 %    imwrite( Final2,str,'tif');
%figure,imshow(Final2)
     
    % % % % % % % % % % /Evaluation Sections
    [m, n] = size(Final2);
    Dimension = m*n;
    Predicted = reshape(Final2, Dimension,1);
    Real = reshape(Truth, Dimension,1);
    TP = sum(Predicted==1 & Real==1);
    TN = sum(Predicted==0 & Real==0);
    FP = sum(Predicted==1 & Real==0);
    FN = sum(Predicted==0 & Real==1);
    Accuracy = (TP + TN)/(TP + TN + FP + FN)
    Acc(counter,1) = Accuracy;
    Sensitivity = TP/(TP + FN)
    Specificity = TN/(TN + FP)
    PPV = TP/(TP + FP)
 end
 [idx idy] = max(Acc)