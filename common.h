#ifndef KINECTFUSION_COMMON_H
#define KINECTFUSION_COMMON_H

#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <fstream>

typedef Eigen::Vector4f Vec4f;
typedef Eigen::Vector3f Vec3f;
typedef Eigen::Vector3i Vec3i;
typedef Eigen::Vector2i Vec2i;
typedef Eigen::Matrix4f Mat4f;
typedef Eigen::Matrix3f Mat3f;
typedef Eigen::Matrix<float, 6, 1> Vec6f;
typedef Eigen::Matrix<double, 6, 1> Vec6d;
typedef Eigen::Matrix<float, 3, 6, Eigen::RowMajor> Mat36f;
typedef Eigen::Matrix<double, 6, 6, Eigen::RowMajor> Mat66d;

#define MAX_WEIGHT 128
#define LEVELS 3

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32

#endif //KINECTFUSION_COMMON_H
