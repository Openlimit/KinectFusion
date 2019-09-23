#include "../common.h"
#include "cuda_common.h"

template<int SIZE>
static __device__ __forceinline__ void reduce(volatile double *buffer) {
    const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    double value = buffer[thread_id];

    if (SIZE >= 1024) {
        if (thread_id < 512) buffer[thread_id] = value = value + buffer[thread_id + 512];
        __syncthreads();
    }
    if (SIZE >= 512) {
        if (thread_id < 256) buffer[thread_id] = value = value + buffer[thread_id + 256];
        __syncthreads();
    }
    if (SIZE >= 256) {
        if (thread_id < 128) buffer[thread_id] = value = value + buffer[thread_id + 128];
        __syncthreads();
    }
    if (SIZE >= 128) {
        if (thread_id < 64) buffer[thread_id] = value = value + buffer[thread_id + 64];
        __syncthreads();
    }

    if (thread_id < 32) {
        if (SIZE >= 64) buffer[thread_id] = value = value + buffer[thread_id + 32];
        if (SIZE >= 32) buffer[thread_id] = value = value + buffer[thread_id + 16];
        if (SIZE >= 16) buffer[thread_id] = value = value + buffer[thread_id + 8];
        if (SIZE >= 8) buffer[thread_id] = value = value + buffer[thread_id + 4];
        if (SIZE >= 4) buffer[thread_id] = value = value + buffer[thread_id + 2];
        if (SIZE >= 2) buffer[thread_id] = value = value + buffer[thread_id + 1];
    }
}

__global__ void estimate_step_kernel(Mat4f cur_pose, Mat4f pre_pose_inv, Mat3f K,
                                     Vec3f *cur_vertex_map, Vec3f *cur_normal_map,
                                     Vec3f *pre_vertex_map, Vec3f *pre_normal_map,
                                     double *global_buffer,
                                     int width, int height, float distance_threshold, float angle_threshold) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    Vec3f n, d, s;
    bool correspondence_found = false;
    if (x < width && y < height) {
        const int idx = y * width + x;
        Vec4f cur_vertex(cur_vertex_map[idx](0), cur_vertex_map[idx](1), cur_vertex_map[idx](2), 1);
        Vec3f cur_normal = cur_normal_map[idx];

        if (!isnan(cur_normal(0)) && (cur_vertex(0) != 0.f || cur_vertex(1) != 0.f || cur_vertex(2) != 0.f)) {
            Vec4f cur_vertex_g4 = cur_pose * cur_vertex;
            Vec4f pre_vertex = pre_pose_inv * cur_vertex_g4;
            Vec3f pre_vertex3 = pre_vertex.head(3);
            Vec3f pre_uvf = K * pre_vertex3 / pre_vertex3(2);

            Vec2i pre_uv(__float2int_rn(pre_uvf(0)), __float2int_rn(pre_uvf(1)));
            if (pre_uv(0) >= 0 && pre_uv(0) < width && pre_uv(1) >= 0 && pre_uv(1) < height && pre_vertex(2) >= 0) {
                int pre_idx = pre_uv(1) * width + pre_uv(0);
                Vec3f pre_vertex_g = pre_vertex_map[pre_idx];
                Vec3f pre_normal_g = pre_normal_map[pre_idx];

                if (!isnan(pre_normal_g(0)) &&
                    (pre_vertex_g(0) != 0.f || pre_vertex_g(1) != 0.f || pre_vertex_g(2) != 0.f)) {
                    Vec3f cur_vertex_g = cur_vertex_g4.head(3);
                    Vec3f res = cur_vertex_g - pre_vertex_g;
                    if (res.norm() <= distance_threshold) {
                        Mat3f cur_rotation = cur_pose.topLeftCorner(3, 3);
                        Vec3f cur_normal_g = cur_rotation * cur_normal;

                        if (cur_normal_g.dot(pre_normal_g) >= angle_threshold) {
                            n = pre_normal_g;
                            d = pre_vertex_g;
                            s = cur_vertex_g;
                            correspondence_found = true;
                        }
                    }
                }
            }
        }
    }

    float row[7];
    if (correspondence_found) {
        ////为啥不取反?
        Vec3f vxt_n = s.cross(n);
        row[0] = vxt_n(0);
        row[1] = vxt_n(1);
        row[2] = vxt_n(2);
        row[3] = n(0);
        row[4] = n(1);
        row[5] = n(2);
        row[6] = n.dot(d - s);
    } else {
        for (int i = 0; i < 7; ++i) {
            row[i] = 0.f;
        }
    }

    __shared__ double smem[BLOCK_SIZE_X * BLOCK_SIZE_Y];
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int bid = blockIdx.y * gridDim.x + blockIdx.x;
    const int grid_size = gridDim.x * gridDim.y;

    int shift = 0;
    for (int i = 0; i < 6; ++i) { // Rows
        for (int j = i; j < 7; ++j) { // Columns and B
            __syncthreads();
            smem[tid] = row[i] * row[j];
            __syncthreads();

            reduce<BLOCK_SIZE_X * BLOCK_SIZE_Y>(smem);

            if (tid == 0) {
                global_buffer[shift * grid_size + bid] = smem[0];
                shift++;
            }
        }
    }
}

__global__ void reduction_kernel(double *global_buffer, int length, double *output) {
    double sum = 0.0;
    for (int t = threadIdx.x; t < length; t += 512)
        sum += global_buffer[blockIdx.x * length + t];

    __shared__ double smem[512];

    smem[threadIdx.x] = sum;
    __syncthreads();

    reduce<512>(smem);

    if (threadIdx.x == 0)
        output[blockIdx.x] = smem[0];
};

void estimate_step(Mat4f &cur_pose, Mat4f &pre_pose, Mat3f &K,
                   Vec3f *cur_vertex_map, Vec3f *cur_normal_map,
                   Vec3f *pre_vertex_map, Vec3f *pre_normal_map,
                   int width, int height, float distance_threshold, float angle_threshold,
                   Mat66d &A, Vec6d &b) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);

    //// A是对称矩阵6X6，只需存21个值，b为6X1
    double *global_buffer;
    CUDA_SAFE_CALL(cudaMalloc((void **) &global_buffer, sizeof(double) * 27 * grid_size.x * grid_size.y));

    estimate_step_kernel << < grid_size, block_size >> > (cur_pose, pre_pose.inverse(), K,
            cur_vertex_map, cur_normal_map, pre_vertex_map, pre_normal_map,
            global_buffer, width, height, distance_threshold, angle_threshold);

    double *sum_buffer;
    CUDA_SAFE_CALL(cudaMalloc((void **) &sum_buffer, sizeof(double) * 27));
    reduction_kernel << < 27, 512 >> > (global_buffer, grid_size.x * grid_size.y, sum_buffer);

    double host_data[27];
    CUDA_SAFE_CALL(cudaMemcpy(host_data, sum_buffer, sizeof(double) * 27, cudaMemcpyDeviceToHost));

    int shift = 0;
    for (int i = 0; i < 6; ++i) { // Rows
        for (int j = i; j < 7; ++j) { // Columns and B
            double value = host_data[shift++];
            if (j == 6)
                b(i) = value;
            else
                A(j, i) = A(i, j) = value;
        }
    }

    CUDA_SAFE_FREE(global_buffer);
    CUDA_SAFE_FREE(sum_buffer);
}