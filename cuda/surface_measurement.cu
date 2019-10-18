#include "../common.h"
#include "cuda_common.h"

__global__ void bilateral_filter_kernel(float *in_data, float *out_data, int width, int height,
                                        int kernel_size, float range_sigma, float spatial_sigma) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height)
        return;

    int idx = y * width + x;
    if (in_data[idx] == 0.f) {
        out_data[idx] = 0.f;
        return;
    }

    int half_kernel_size = kernel_size / 2;
    int min_x = fmaxf(x - half_kernel_size, 0);
    int max_x = fminf(x + half_kernel_size, width - 1);
    int min_y = fmaxf(y - half_kernel_size, 0);
    int max_y = fminf(y + half_kernel_size, height - 1);

    float weight_sum = 0;
    float D = 0;
    for (int u = min_x; u <= max_x; ++u) {
        for (int v = min_y; v <= max_y; ++v) {
            int cur_idx = v * width + u;
            if (in_data[cur_idx] == 0.f)
                continue;

            float spatial = (x - u) * (x - u) + (y - v) * (y - v);
            float range = (in_data[idx] - in_data[cur_idx]) * (in_data[idx] - in_data[cur_idx]);
            float weight = expf(-spatial / (spatial_sigma * spatial_sigma)) *
                           expf(-range / (range_sigma * range_sigma));
            D += weight * in_data[cur_idx];
            weight_sum += weight;
        }
    }

    out_data[idx] = D / weight_sum;
}

__global__ void compute_vertex_kernel(float *data, Vec3f *vertex_map, Mat3f K_inverse, int width, int height) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height)
        return;

    int idx = y * width + x;
    Vec3f u(x, y, 1);
    vertex_map[idx] = data[idx] * K_inverse * u;
}

__global__ void compute_normal_kernel(Vec3f *vertex_map, Vec3f *normal_map, int width, int height) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height)
        return;

    int idx = y * width + x;
    if (x == 0 || x + 1 == width || y == 0 || y + 1 == height) {
        normal_map[idx].setConstant(FLT_MAX);
        return;
    }

    Vec3f left = vertex_map[y * width + x - 1];
    Vec3f right = vertex_map[y * width + x + 1];
    Vec3f upper = vertex_map[(y - 1) * width + x];
    Vec3f lower = vertex_map[(y + 1) * width + x];

    Vec3f normal;
    if (left(2) == 0 || right(2) == 0 || upper(2) == 0 || lower(2) == 0)
        normal = Vec3f(0.f, 0.f, 0.f);
    else {
        Vec3f hor(left(0) - right(0), left(1) - right(1), left(2) - right(2));
        Vec3f ver(upper(0) - lower(0), upper(1) - lower(1), upper(2) - lower(2));

        normal = hor.cross(ver);
        normal.normalize();

        ////取负号是因为表面法向是朝着z轴负方向的
        if (normal(2) > 0)
            normal *= -1;
    }

    normal_map[idx] = normal;
}

__global__ void pyrDown_kernel(float *in_data, float *out_data, int out_width, int out_height,
                               int kernel_size, float range_sigma) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= out_width || y >= out_height)
        return;

    int in_width = out_width * 2;
    int in_height = out_height * 2;
    int in_x = x * 2;
    int in_y = y * 2;
    int in_idx = in_y * in_width + in_x;

    int idx = y * out_width + x;
    if (in_data[in_idx] == 0.f) {
        out_data[idx] = 0.f;
        return;
    }

    int half_kernel_size = kernel_size / 2;
    int min_x = fmaxf(in_x - half_kernel_size, 0);
    int max_x = fminf(in_x + half_kernel_size, in_width - 1);
    int min_y = fmaxf(in_y - half_kernel_size, 0);
    int max_y = fminf(in_y + half_kernel_size, in_height - 1);

    float weight = 0;
    float D = 0;
    for (int u = min_x; u <= max_x; ++u) {
        for (int v = min_y; v <= max_y; ++v) {
            int cur_idx = v * in_width + u;
            if (abs(in_data[in_idx] - in_data[cur_idx]) > range_sigma * 3)
                continue;
            D += in_data[cur_idx];
            weight += 1;
        }
    }

    out_data[idx] = D / weight;
}

__global__ void downsample_kernel(float *in_data, float *out_data, int out_width, int out_height, int down_factor) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= out_width || y >= out_height)
        return;

    int in_width = out_width * down_factor;
    int in_x = x * down_factor;
    int in_y = y * down_factor;
    int in_idx = in_y * in_width + in_x;
    int idx = y * out_width + x;
    out_data[idx] = in_data[in_idx];
}

__global__ void compute_gradient_kernel(float *data, Vec2f *gradient, int width, int height) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height)
        return;
    int idx = y * width + x;
    if (x + 1 == width || y + 1 == height) {
        gradient[idx].setZero();
    }

    float gx = data[y * width + x + 1] - data[idx];
    float gy = data[(y + 1) * width + x] - data[idx];
    gradient[idx] = Vec2f(gx, gy);
}

void bilateral_filter(float *in_data, float *out_data, int width, int height,
                      int kernel_size, float range_sigma, float spatial_sigma) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);
    bilateral_filter_kernel << < grid_size, block_size >> > (in_data, out_data, width, height,
            kernel_size, range_sigma, spatial_sigma);
    cudaDeviceSynchronize();
}

void compute_vertex_map(float *data, Vec3f *vertex_map, Mat3f &K, int width, int height) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);
    compute_vertex_kernel << < grid_size, block_size >> > (data, vertex_map, K.inverse(), width, height);
    cudaDeviceSynchronize();
}

void compute_normal_map(Vec3f *vertex_map, Vec3f *normal_map, int width, int height) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);
    compute_normal_kernel << < grid_size, block_size >> > (vertex_map, normal_map, width, height);
    cudaDeviceSynchronize();
}

void pyrDown(float *in_data, float *out_data, int out_width, int out_height, int kernel_size, float range_sigma) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((out_width + block_size.x - 1) / block_size.x, (out_height + block_size.y - 1) / block_size.y);
    pyrDown_kernel << < grid_size, block_size >> > (in_data, out_data, out_width, out_height, kernel_size, range_sigma);
    cudaDeviceSynchronize();
}

void downsample(float *in_data, float *out_data, int out_width, int out_height, int down_factor) {
    dim3 block_size(8, 8);
    dim3 grid_size((out_width + block_size.x - 1) / block_size.x, (out_height + block_size.y - 1) / block_size.y);
    downsample_kernel << < grid_size, block_size >> > (in_data, out_data, out_width, out_height, down_factor);
    cudaDeviceSynchronize();
}

void compute_gradient(float *data, Vec2f *gradient, int width, int height) {
    dim3 block_size(8, 8);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);
    compute_gradient_kernel << < grid_size, block_size >> > (data, gradient, width, height);
    cudaDeviceSynchronize();
}