cmake_minimum_required(VERSION 3.10)
project(KinectFusion)

set(CMAKE_CXX_STANDARD 11)

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

CUDA_ADD_EXECUTABLE(KinectFusion common.h cuda/cuda_common.h cuda/surface_prediction.cu
        cuda/surface_measurement.cu cuda/pose_estimate.cu cuda/surface_reconstruction.cu
         KinectFusion.h KinectFusion.cpp main.cpp)