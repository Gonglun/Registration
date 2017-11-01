clc;
clear all;
close all;
SIZEX=256;
SIZEY=256;
SIZEZ=106;

fid=fopen('C:\Users\Administrator\Desktop\MI\\Test_image.txt','rb');
[out,count]=fread(fid,SIZEX*SIZEY*SIZEZ,'float');
out=reshape(out,SIZEX,SIZEY,SIZEZ);
Transform=out;
imageView3d(Transform);