function gb=gabor_fn(bw,gamma,psi,lambda,theta)
 
sigma = lambda/pi*sqrt(log(2)/2)*(2^bw+1)/(2^bw-1);
sigma_x = sigma;
sigma_y = 2.5 * sigma/gamma;

sz=fix(2*max(sigma_y,sigma_x));
if mod(sz,2)==0, sz=sz+1;end
[x y]=meshgrid(-fix(sz/2):fix(sz/2),fix(sz/2):-1:fix(-sz/2));
f = 2*pi/lambda;
b = 1/(2*sigma^2);
a = b/pi;
x_theta=x*cos(theta)+y*sin(theta);
y_theta=-x*sin(theta)+y*cos(theta);
 cosfunc = cos(f * x_theta - pi);
gb= a * exp(-b*(x_theta.^2 + (gamma^2 .* y_theta.^2))).*cosfunc;
