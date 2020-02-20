function [lp]= createmask(ImCrop)
[ir,ic,iz] = size(ImCrop);
hr = (ir-1)/2; 
hc = (ic-1)/2; 
[x, y] = meshgrid(-hc:hc, -hr:hr);
mg = sqrt((x/hc).^2 + (y/hr).^2); 
lp = double(mg <= (1- 0.06));
% imwrite(lp,'1.tif','tif');
% size(lp)
% Final1 = lp - FinalImage;
% figure,imshow(Final1)

end