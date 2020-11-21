clear;
clc;

%Step1:��ȡͼ�񣬲���Ϊ�Ҷ�ͼ
%��ȡͼ��
img = imread('test.jpg');
%��ȡRGB����
R = img(:, :, 1);
G = img(:, :, 2);
B = img(:, :, 3);
%ת��Ϊ�Ҷ�ͼ������rgb��Ȩ��
gimg = 0.299 * R + 0.587 * G + 0.114 * B;
%gimg = rgb2gray(img);
imshow(gimg);
title('�Ҷ�ͼ');

%Step2 �Ҷ�ֱ��ͼ
[m, n] = size(gimg);
%������ֵ�Ƶ��
frequency = zeros(1,256);
for i = 1:m
	for j = 1:n
        frequency(1, gimg(i,j)+1) = frequency(1, gimg(i,j)+1)+1;
    end
end
%������ֵ�Ƶ��
probability = frequency / (m * n);
figure,bar(0:255,probability,'k');
title('ԭͼ��ֱ��ͼ')%��ʾԭֱ��ͼ������
xlabel('�Ҷ�ֵ')
ylabel('����Ƶ��')

%Step3 ��ɢ����Ҷ�任����Ƶ��ͼ
%ʹ�ø���Ҷ�任�Ŀɷ�����
%��ȡͼƬ��С
%������˾���
vy = (0: m - 1)' * (0: m - 1);
M_vy = exp(-2*1i*pi* vy / m);
%�����ҳ˾���
ux = (0: n-1)' * (0: n-1);
M_ux = exp(-2*1i*pi*ux/n);

%��ɢ����Ҷ�任
F = M_vy * double(gimg) * M_ux;
% ���Ļ�
Fc = fftshift(F); 
% ȡģ
Fm = abs(Fc); 
%ȡ������ʾ
Fm = log(Fm);
figure, imshow(Fm,[]);
title('��ɢ����Ҷ�任Ƶ�׷���ͼ');

%Step4 ֱ��ͼ���⻯
%�����ۻ�ֱ��ͼ
accumulate = zeros(1, 256);
for i = 1:256
    for j = 1:i
        accumulate(1,i) = accumulate(1,i) + probability(1,j);
    end
end
%ȡ����ӳ���ϵ�洢�ڸþ����У�ԭͼ��Ҷȼ�i(0-255)->�µĻҶȼ�accumulate(1,i+1)
accumulate = round((accumulate * 255) + 0.5);

%��������ĻҶ�ֱ��ͼ
L = zeros(1, 256);
for i = 1:256
    L(accumulate(1,i)) = L(accumulate(1,i)) + probability(1,i) ;
end
figure,bar(0:255,L,'k'),title('���⻯��Ҷ�ֱ��ͼ');
xlabel('�Ҷ�ֵ'),ylabel('����Ƶ��')

%��ԭͼ��ӳ��
Trans = gimg;
for i = 1:m
    for j =1:n
        Trans(i,j) = accumulate(Trans(i,j)+1);
    end
end
figure,imshow(Trans);
title('ֱ��ͼ���⻯��ͼ��')


%Step5 ̬ͬ�˲�
I=double(gimg);
%��Ƶ����
rL=0.2;
%��Ƶ����
rH=5.0;
%�˲�����������ϵ��
c=2;
%��ֹƵ��
d0=10;
%ȡ����
I1=log(I+1);
%����Ҷ�任
FI=fft2(I1);
n1=floor(m/2);
n2=floor(n/2);
for i=1:m
    for j=1:n
        D(i,j)=((i-n1).^2+(j-n2).^2);
        %��˹̬ͬ�˲�
        H(i,j)=(rH-rL).*(exp(c*(-D(i,j)./(d0^2))))+rL;
    end
end
%����Ҷ��任
I2=ifft2(H.*FI);
%ȡָ������
I3=real(exp(I2));
figure;imshow(I3,[]);title('̬ͬ�˲���ǿ��')