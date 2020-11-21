clear;
clc;

%Step1:读取图像，并变为灰度图
%读取图像
img = imread('test.jpg');
%读取RGB矩阵
R = img(:, :, 1);
G = img(:, :, 2);
B = img(:, :, 3);
%转化为灰度图，采用rgb加权法
gimg = 0.299 * R + 0.587 * G + 0.114 * B;
%gimg = rgb2gray(img);
imshow(gimg);
title('灰度图');

%Step2 灰度直方图
[m, n] = size(gimg);
%计算出现的频次
frequency = zeros(1,256);
for i = 1:m
	for j = 1:n
        frequency(1, gimg(i,j)+1) = frequency(1, gimg(i,j)+1)+1;
    end
end
%计算出现的频率
probability = frequency / (m * n);
figure,bar(0:255,probability,'k');
title('原图像直方图')%显示原直方图的数据
xlabel('灰度值')
ylabel('出现频率')

%Step3 离散傅里叶变换幅度频谱图
%使用傅里叶变换的可分离性
%获取图片大小
%定义左乘矩阵
vy = (0: m - 1)' * (0: m - 1);
M_vy = exp(-2*1i*pi* vy / m);
%定义右乘矩阵
ux = (0: n-1)' * (0: n-1);
M_ux = exp(-2*1i*pi*ux/n);

%离散傅里叶变换
F = M_vy * double(gimg) * M_ux;
% 中心化
Fc = fftshift(F); 
% 取模
Fm = abs(Fc); 
%取对数显示
Fm = log(Fm);
figure, imshow(Fm,[]);
title('离散傅里叶变换频谱幅度图');

%Step4 直方图均衡化
%计算累积直方图
accumulate = zeros(1, 256);
for i = 1:256
    for j = 1:i
        accumulate(1,i) = accumulate(1,i) + probability(1,j);
    end
end
%取整，映射关系存储在该矩阵中，原图像灰度级i(0-255)->新的灰度级accumulate(1,i+1)
accumulate = round((accumulate * 255) + 0.5);

%计算均衡后的灰度直方图
L = zeros(1, 256);
for i = 1:256
    L(accumulate(1,i)) = L(accumulate(1,i)) + probability(1,i) ;
end
figure,bar(0:255,L,'k'),title('均衡化后灰度直方图');
xlabel('灰度值'),ylabel('出现频率')

%将原图像映射
Trans = gimg;
for i = 1:m
    for j =1:n
        Trans(i,j) = accumulate(Trans(i,j)+1);
    end
end
figure,imshow(Trans);
title('直方图均衡化后图像')


%Step5 同态滤波
I=double(gimg);
%低频增益
rL=0.2;
%高频增益
rH=5.0;
%滤波器函数的锐化系数
c=2;
%截止频率
d0=10;
%取对数
I1=log(I+1);
%傅里叶变换
FI=fft2(I1);
n1=floor(m/2);
n2=floor(n/2);
for i=1:m
    for j=1:n
        D(i,j)=((i-n1).^2+(j-n2).^2);
        %高斯同态滤波
        H(i,j)=(rH-rL).*(exp(c*(-D(i,j)./(d0^2))))+rL;
    end
end
%傅里叶逆变换
I2=ifft2(H.*FI);
%取指数操作
I3=real(exp(I2));
figure;imshow(I3,[]);title('同态滤波增强后')