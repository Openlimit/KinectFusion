//
// Created by meidai on 19-9-19.
//

#ifndef KINECTFUSION_CUDA_COMMON_H
#define KINECTFUSION_CUDA_COMMON_H

//#ifndef __CUDACC__
//#define __CUDACC__
//#endif

#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32

#define CUDA_SAFE_CALL(b) { if(b != cudaSuccess) {std::cout<< std::string(cudaGetErrorString(b)) <<std::endl;throw std::string(cudaGetErrorString(b));} }
#define CUDA_SAFE_FREE(b) { if(b) { CUDA_SAFE_CALL(cudaFree(b)); b = NULL; } }

#endif //KINECTFUSION_CUDA_COMMON_H
