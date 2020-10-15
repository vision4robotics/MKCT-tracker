# MKCT-tracker

| **Test passed**                                              |
| ------------------------------------------------------------ |
| [![matlab-2017b](https://img.shields.io/badge/matlab-2017b-yellow.svg)](https://www.mathworks.com/products/matlab.html) [![MatConvNet-1.0--beta25](https://img.shields.io/badge/MatConvNet-1.0--beta25%20-blue.svg)](http://www.vlfeat.org/matconvnet/download/matconvnet-1.0-beta25.tar.gz) ![CUDA-8.0](https://img.shields.io/badge/CUDA-8.0-green.svg) |

> Matlab implementation of *Robust multi-kernelized correlators for UAV tracking with adaptive context analysis and dynamic weighted filters* (MKCT-tracker).

## Publication and Citation

This paper has been published by Neural Computing and Applications.

You can find this paper here: https://link.springer.com/article/10.1007/s00521-020-04716-x.

Please cite it as:

@article{Fu2020Robust, 

title={Robust multi-kernelized correlators for UAV tracking with adaptive context analysis and dynamic weighted filters}, 

author={Fu, Changhong and He, Yujie and Lin Fuling and Xiong Weijiang}, 

journal={Neural Computing and Applications}, 

year={2020} 

}

## Results

The following are the results from the experiment conducted on 100 challenging sequences extracted from UAV123@10fps.

![Prec](./result/Prec.png)

![Succ](./result/Succ.png)

## Instructions

1. Enter `/model` folder and run `vgg_19_dl.m` to download `imagenet-vgg-verydeep-19.mat` from [here](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat).
2. Download matconvnet toolbox [here](http://www.vlfeat.org/matconvnet/download/matconvnet-1.0-beta25.tar.gz) and put it in `/external`.
3. Run `MKCT_Demo.m`

Note: the default demo is using CPU to run the whole program. If GPU is required, just change `false` in the following lines in `MKCT_Demo.m` to `true`:

```
vl_compilenn('enableGpu',true,... 
             'cudaRoot', '<your-cuda-root>' ...
            );
vl_setupnn();

global enableGPU;
enableGPU = false;
```

## Acknowledgements

Partly borrowed from [KCC](https://github.com/wang-chen/KCC/tree/master/tracking).
