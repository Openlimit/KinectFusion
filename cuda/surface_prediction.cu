#include "../common.h"
#include "cuda_common.h"

__device__ __forceinline__
float interpolate_trilinearly(Vec3f &point, float2 *volume, int3 &volume_size) {
    Vec3i point_in_grid = point.cast<int>();

    float vx = point_in_grid(0) + 0.5f;
    float vy = point_in_grid(1) + 0.5f;
    float vz = point_in_grid(2) + 0.5f;

    point_in_grid(0) = (point(0) < vx) ? (point_in_grid(0) - 1) : point_in_grid(0);
    point_in_grid(1) = (point(1) < vy) ? (point_in_grid(1) - 1) : point_in_grid(1);
    point_in_grid(2) = (point(2) < vz) ? (point_in_grid(2) - 1) : point_in_grid(2);

    const float a = point(0) - (point_in_grid(0) + 0.5f);
    const float b = point(1) - (point_in_grid(1) + 0.5f);
    const float c = point(2) - (point_in_grid(2) + 0.5f);

    int yz_size = volume_size.y * volume_size.z;
    return volume[point_in_grid(0) * yz_size + point_in_grid(1) * volume_size.z +
                  point_in_grid(2)].x * (1 - a) * (1 - b) * (1 - c) +
           volume[point_in_grid(0) * yz_size + point_in_grid(1) * volume_size.z +
                  point_in_grid(2) + 1].x * (1 - a) * (1 - b) * c +
           volume[point_in_grid(0) * yz_size + (point_in_grid(1) + 1) * volume_size.z +
                  point_in_grid(2)].x * (1 - a) * b * (1 - c) +
           volume[point_in_grid(0) * yz_size + (point_in_grid(1) + 1) * volume_size.z +
                  point_in_grid(2) + 1].x * (1 - a) * b * c +
           volume[(point_in_grid(0) + 1) * yz_size + point_in_grid(1) * volume_size.z +
                  point_in_grid(2)].x * a * (1 - b) * (1 - c) +
           volume[(point_in_grid(0) + 1) * yz_size + point_in_grid(1) * volume_size.z +
                  point_in_grid(2) + 1].x * a * (1 - b) * c +
           volume[(point_in_grid(0) + 1) * yz_size + (point_in_grid(1) + 1) * volume_size.z +
                  point_in_grid(2)].x * a * b * (1 - c) +
           volume[(point_in_grid(0) + 1) * yz_size + (point_in_grid(1) + 1) * volume_size.z +
                  point_in_grid(2) + 1].x * a * b * c;
}


__device__ __forceinline__
float get_min_time(Vec3f &volume_max, Vec3f &volume_min, Vec3f &origin, Vec3f &direction) {
    float txmin = ((direction(0) > 0 ? volume_min(0) : volume_max(0)) - origin(0)) / direction(0);
    float tymin = ((direction(1) > 0 ? volume_min(1) : volume_max(1)) - origin(1)) / direction(1);
    float tzmin = ((direction(2) > 0 ? volume_min(2) : volume_max(2)) - origin(2)) / direction(2);

    return fmaxf(fmaxf(txmin, tymin), tzmin);
}

__device__ __forceinline__
float get_max_time(Vec3f &volume_max, Vec3f &volume_min, Vec3f &origin, Vec3f &direction) {
    float txmax = ((direction(0) > 0 ? volume_max(0) : volume_min(0)) - origin(0)) / direction(0);
    float tymax = ((direction(1) > 0 ? volume_max(1) : volume_min(1)) - origin(1)) / direction(1);
    float tzmax = ((direction(2) > 0 ? volume_max(2) : volume_min(2)) - origin(2)) / direction(2);

    return fminf(fminf(txmax, tymax), tzmax);
}

__global__ void raycast_tsdf_kernel(float2 *tsdf_volume, Vec3f *vertex_map, Vec3f *normal_map,
                                    int width, int height, float truncation_distance,
                                    int3 volume_size, float voxel_scale, Vec3f volume_origin,
                                    Mat3f K_inv, Mat3f rotation, Vec3f translation) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;
    vertex_map[y * width + x].setZero();
    normal_map[y * width + x].setZero();

    Vec3f volume_max(volume_size.x * voxel_scale + volume_origin(0),
                     volume_size.y * voxel_scale + volume_origin(1),
                     volume_size.z * voxel_scale + volume_origin(2));

    Vec3f uv(x, y, 1.f);
    Vec3f pixel_position = K_inv * uv;

    ////使光线方向变为全局坐标系下的方向
    Vec3f ray_direction = rotation * pixel_position;
    ray_direction.normalize();

    //// 计算光线与AABB交点
    float ray_length = fmaxf(get_min_time(volume_max, volume_origin, translation, ray_direction), 0.f);
    if (ray_length >= get_max_time(volume_max, volume_origin, translation, ray_direction))
        return;

    //// 计算volume内局部坐标
    ray_length += voxel_scale;
    Vec3f grid = (translation + ray_direction * ray_length - volume_origin) / voxel_scale;

    //// 一定记得不要越界，不然会报错:an illegal memory access was encountered
    if (grid(0) < 0 || grid(0) >= volume_size.x ||
        grid(1) < 0 || grid(1) >= volume_size.y ||
        grid(2) < 0 || grid(2) >= volume_size.z)
        return;

    float tsdf = tsdf_volume[__float2int_rd(grid(0)) * volume_size.y * volume_size.z +
                             __float2int_rd(grid(1)) * volume_size.z
                             + __float2int_rd(grid(2))].x;

    float max_search_length = ray_length + volume_size.x * voxel_scale * sqrtf(2.f);
    for (; ray_length < max_search_length; ray_length += truncation_distance * 0.5f) {
        grid = (translation + ray_direction * (ray_length + truncation_distance * 0.5f) - volume_origin) / voxel_scale;

        if (grid(0) < 1 || grid(0) >= volume_size.x - 1 ||
            grid(1) < 1 || grid(1) >= volume_size.y - 1 ||
            grid(2) < 1 || grid(2) >= volume_size.z - 1)
            continue;

        float previous_tsdf = tsdf;
        tsdf = tsdf_volume[__float2int_rd(grid(0)) * volume_size.y * volume_size.z +
                           __float2int_rd(grid(1)) * volume_size.z
                           + __float2int_rd(grid(2))].x;

        if (previous_tsdf < 0.f && tsdf > 0.f) //Zero crossing from behind
            break;
        if (previous_tsdf > 0.f && tsdf < 0.f) { //Zero crossing
            float t_star = ray_length - truncation_distance * 0.5f * previous_tsdf / (tsdf - previous_tsdf);

            Vec3f vertex = translation + ray_direction * t_star;

            Vec3f location_in_grid = (vertex - volume_origin) / voxel_scale;
            if (location_in_grid(0) < 1 | location_in_grid(0) >= volume_size.x - 1 ||
                location_in_grid(1) < 1 || location_in_grid(1) >= volume_size.y - 1 ||
                location_in_grid(2) < 1 || location_in_grid(2) >= volume_size.z - 1)
                break;

            Vec3f normal, shifted;

            shifted = location_in_grid;
            shifted(0) += 1;
            if (shifted(0) >= volume_size.x - 1)
                break;
            float Fx1 = interpolate_trilinearly(shifted, tsdf_volume, volume_size);

            shifted = location_in_grid;
            shifted(0) -= 1;
            if (shifted(0) < 1)
                break;
            float Fx2 = interpolate_trilinearly(shifted, tsdf_volume, volume_size);

            normal(0) = Fx1 - Fx2;

            shifted = location_in_grid;
            shifted(1) += 1;
            if (shifted(1) >= volume_size.y - 1)
                break;
            float Fy1 = interpolate_trilinearly(shifted, tsdf_volume, volume_size);

            shifted = location_in_grid;
            shifted(1) -= 1;
            if (shifted(1) < 1)
                break;
            float Fy2 = interpolate_trilinearly(shifted, tsdf_volume, volume_size);

            normal(1) = Fy1 - Fy2;

            shifted = location_in_grid;
            shifted(2) += 1;
            if (shifted(2) >= volume_size.z - 1)
                break;
            float Fz1 = interpolate_trilinearly(shifted, tsdf_volume, volume_size);

            shifted = location_in_grid;
            shifted(2) -= 1;
            if (shifted(2) < 1)
                break;
            float Fz2 = interpolate_trilinearly(shifted, tsdf_volume, volume_size);

            normal(2) = Fz1 - Fz2;

            if (normal.norm() == 0)
                break;

            normal.normalize();

            vertex_map[y * width + x] = vertex;
            normal_map[y * width + x] = normal;
            break;
        }
    }
}

void raycast_tsdf(float2 *tsdf_volume, Vec3f *vertex_map, Vec3f *normal_map,
                  int width, int height, float truncation_distance,
                  int3 &volume_size, float voxel_scale, Vec3f &volume_origin,
                  Mat3f &K, Mat4f &pose) {
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);
    raycast_tsdf_kernel << < grid_size, block_size >> > (tsdf_volume, vertex_map, normal_map, width, height,
            truncation_distance, volume_size, voxel_scale, volume_origin,
            K.inverse(), pose.block(0, 0, 3, 3), pose.block(0, 3, 3, 1));
    cudaDeviceSynchronize();
}