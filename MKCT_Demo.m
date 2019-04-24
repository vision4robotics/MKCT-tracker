%%% Note that the default setting is CPU. TO ENABLE GPU, please recompile the MatConvNet toolbox
function MKCT_Demo(~)
close all;
clear;
clc;    
setup_paths();
% complie with your CUDA path `vl_compilenn('enableGpu',true);`
% vl_compilenn('enableGpu',true,... 
%              'cudaRoot', '<your-cuda-root>' ...
%              );
vl_setupnn();
global enableGPU;
enableGPU = false;
% Load video information
seq = load_video_information('UAV123_10fps');
result  =  run_MKCT(seq);