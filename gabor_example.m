function img_out_disp = gabor_example(Img,lambda,theta,psi,gamma,bw,N)
img_in = Img;
img_out = zeros(size(img_in,1), size(img_in,2), N);
for n=1:N
    gb = gabor_fn(bw,gamma,psi(1),lambda,theta)...
    +  gabor_fn(bw,gamma,psi(2),lambda,theta);
    img_out(:,:,n) = imfilter(img_in, gb, 'symmetric');
    theta = theta + 2*pi/N;
end
img_out_disp = sum(abs(img_out).^2, 3).^0.5;