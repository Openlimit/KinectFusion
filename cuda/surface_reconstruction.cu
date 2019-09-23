#include <device_functions.h>
#include "../common.h"
#include "cuda_common.h"

__global__ void update_tsdf_kernel(float *depth_map, float2 *tsdf_volume, Vec3f volume_origin,
                                   int width, int height,
                                   int3 volume_size, float voxel_scale,
                                   float truncation_distance, Mat4f pose_inv, Mat3f K, Mat3f K_inv) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= volume_size.x || y >= volume_size.y)
        return;

    const int idx = x * volume_size.y * volume_size.z + y * volume_size.z;
    for (int z = 0; z < volume_size.z; ++z) {
        Vec4f position((x + 0.5f) * voxel_scale + volume_origin(0),
                       (y + 0.5f) * voxel_scale + volume_origin(1),
                       (z + 0.5f) * voxel_scale + volume_origin(2), 1);
        Vec4f camera_pos = pose_inv * position;

        if (camera_pos(2) <= 0)
            continue;

        Vec3f camera_pos3 = camera_pos.head(3);
        Vec3f uvf = K * camera_pos3 / camera_pos(2);
        Vec2i uv(__float2int_rn(uvf(0)), __float2int_rn(uvf(1)));

        if (uv(0) < 0 || uv(0) >= width || uv(1) < 0 || uv(1) >= height)
            continue;

        float depth = depth_map[uv(1) * width + uv(0)];
        if (depth <= 0)
            continue;

        Vec3f uv3(uv(0), uv(1), 1.f);
        Vec3f xylambda = K_inv * uv3;
        float lambda = xylambda.norm();

        //// 可视空间z值比zero-cross小，所以取反
        float sdf = -1.f * ((1.f / lambda) * camera_pos.norm() - depth);

        if (sdf >= -truncation_distance) {
            float new_tsdf = fminf(1.f, sdf / truncation_distance);

            float2 voxel_tuple = tsdf_volume[idx + z];

            float current_tsdf = voxel_tuple.x;
            float current_weight = voxel_tuple.y;

            float add_weight = 1;
            float updated_tsdf =
                    (current_weight * current_tsdf + add_weight * new_tsdf) / (current_weight + add_weight);
            float updated_weight = fminf(current_weight + add_weight, MAX_WEIGHT);

            tsdf_volume[idx + z] = make_float2(updated_tsdf, updated_weight);
        }
    }
}

void update_tsdf(float *depth_map, float2 *tsdf_volume, Vec3f &volume_origin,
                 int width, int height,
                 int3 volume_size, float voxel_scale,
                 float truncation_distance, Mat4f &pose, Mat3f &K) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((volume_size.x + block_size.x - 1) / block_size.x,
                   (volume_size.y + block_size.y - 1) / block_size.y);
    update_tsdf_kernel << < grid_size, block_size >> > (depth_map, tsdf_volume, volume_origin, width, height,
            volume_size, voxel_scale, truncation_distance, pose.inverse(), K, K.inverse());
    cudaDeviceSynchronize();
}