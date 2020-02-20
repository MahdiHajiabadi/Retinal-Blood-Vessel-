function Feature = Preprocessing(im,counter)
    ImCrop = im(30:564, 15:550,1:3);
     [lp] = createmask(ImCrop);
%      str  = strcat('ImCrop',num2str(counter),'.tif');
%     imwrite(ImCrop,str,'tif');
    % imshow(ImCrop);
    % % % =========================== Trnasformation Phase 
    %%=============================1: Lab transformation 
     colorTransform = makecform('srgb2lab');
    lab = applycform(ImCrop,colorTransform);
%       str  = strcat('lab',num2str(counter),'.tif');
%     imwrite(lab,str,'tif');
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
    %%%% ================================= Final Trnasformation Image
    GRB=ImCrop(:,:,2);
%      str  = strcat('GRB',num2str(counter),'.tif');
%     imwrite(GRB,str,'tif');
    F = YCbCr(:,:,1);
%     str  = strcat('YCbCr_',num2str(counter),'.tif');
%     imwrite(F,str,'tif');
    D = lab(:,:,1);
%     str  = strcat('ColorTransform_',num2str(counter),'.tif');
%     imwrite(D,str,'tif');
    G = GImage(:,:,1);
%     str  = strcat('Gaussian_',num2str(counter),'.tif');
%     imwrite(G,str,'tif');
    % imshow(IMF)
    %%%=================== CLAHE (Contrast-limited Adaptive Histogram Equalization) algorithm for contrast Enhancement
    % imshow(GrayScale);
    Green =  adapthisteq(GRB,'clipLimit',0.01,'Distribution','uniform');
%     str  = strcat('CLAHE_Green',num2str(counter),'.tif');
%     imwrite(Green,str,'tif');
    Y =  adapthisteq(F,'clipLimit',0.01,'Distribution','uniform');
%     str  = strcat('CLAHE_YCbCr',num2str(counter),'.tif');
%     imwrite(Y,str,'tif');
    L =  adapthisteq(D,'clipLimit',0.01,'Distribution','uniform');
%     str  = strcat('CLAHE_Lab',num2str(counter),'.tif');
%     imwrite(L,str,'tif');
    G1 =  adapthisteq(G,'clipLimit',0.01,'Distribution','uniform');
%     str  = strcat('CLAHE_Gaussian',num2str(counter),'.tif');
%     imwrite(G1,str,'tif');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Gabor Filtering on the 4 Channels (Green,Y,L,G1 )
    Input = cell(4,1);
    Input{1} = G;
    Input{2} = G1;
    Input{3} = Y;
    Input{4} = Green;
    theta   = 0;    %%%% theta -> angle in rad
    lambda  = [9 10 11];   %%%%% lambda -> Wave Length
    psi     = [-pi pi]; %%% psi ->phase shift
    gamma   = 0.5; %%%%% gamma ->aspect ration
    bw      = 1;   %%% bw -> bandwidth
    N       = 24;
    % % % 
    Feature = zeros(size(G1,1),size(G1,2),13);
    for i =1:4
        for j = 1:3
        KHSH  = Input{i};
       GG_example = gabor_example(KHSH,lambda(j),theta,psi,gamma,bw,N);
    %     imshow(GG_example);
       maxIntensify = max(max(GG_example));
       perVal = maxIntensify * .1;
       GG_example(GG_example<=perVal)=0;
       GG_example(GG_example>perVal)=1;
%        str  = strcat('Gabor_',num2str(counter),num2str(i),num2str(j),'.tif');
%        imwrite(GG_example,str,'tif');
    %    figure
    %   
     GG_final = (lp .*(GG_example));
%     figure,imshow(GG_final);
       Feature(:,:,(i-1)*3 + j) = GG_final;
    %    imshow(GG_example)

    
        end
    end
    Green = double(Green)/255;
    Feature(:,:,13) = Green;
end
