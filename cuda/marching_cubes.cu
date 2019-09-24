#include "../common.h"
#include "cuda_common.h"

__device__ int global_count = 0;

//##### HELPERS #####
static __device__ __forceinline__
unsigned int lane_ID() {
    unsigned int ret;
    asm("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}

static __device__ __forceinline__
int laneMaskLt() {
    unsigned int ret;
    asm("mov.u32 %0, %lanemask_lt;" : "=r"(ret));
    return ret;
}

static __device__ __forceinline__
int binaryExclScan(int ballot_mask) {
    return __popc(laneMaskLt() & ballot_mask);
}

__device__ __forceinline__
float read_tsdf(float2 *tsdf_volume, int3 &volume_size, int x, int y, int z, float &weight) {
    float2 v = tsdf_volume[x * volume_size.y * volume_size.z + y * volume_size.z + z];
    weight = v.y;
    return v.x;
}

__device__ __forceinline__
int compute_cube_index(float2 *tsdf_volume, int3 &volume_size,
                       int x, int y, int z, float *tsdf_values) {
    float weight;
    int cube_index = 0; // calculate flag indicating if each vertex is inside or outside isosurface

    //// 注意赋值符号和不等号的优先级，这里需要加括号
    cube_index += static_cast<int>((tsdf_values[0] = read_tsdf(tsdf_volume, volume_size, x, y, z, weight)) < 0.f);
    if (weight == 0) return 0;
    cube_index +=
            static_cast<int>((tsdf_values[1] = read_tsdf(tsdf_volume, volume_size, x + 1, y, z, weight)) < 0.f) << 1;
    if (weight == 0) return 0;
    cube_index +=
            static_cast<int>((tsdf_values[2] = read_tsdf(tsdf_volume, volume_size, x + 1, y + 1, z, weight)) < 0.f)
                    << 2;
    if (weight == 0) return 0;
    cube_index +=
            static_cast<int>((tsdf_values[3] = read_tsdf(tsdf_volume, volume_size, x, y + 1, z, weight)) < 0.f) << 3;
    if (weight == 0) return 0;
    cube_index +=
            static_cast<int>((tsdf_values[4] = read_tsdf(tsdf_volume, volume_size, x, y, z + 1, weight)) < 0.f) << 4;
    if (weight == 0) return 0;
    cube_index +=
            static_cast<int>((tsdf_values[5] = read_tsdf(tsdf_volume, volume_size, x + 1, y, z + 1, weight)) < 0.f)
                    << 5;
    if (weight == 0) return 0;
    cube_index +=
            static_cast<int>((tsdf_values[6] = read_tsdf(tsdf_volume, volume_size, x + 1, y + 1, z + 1, weight)) < 0.f)
                    << 6;
    if (weight == 0) return 0;
    cube_index +=
            static_cast<int>((tsdf_values[7] = read_tsdf(tsdf_volume, volume_size, x, y + 1, z + 1, weight)) < 0.f)
                    << 7;
    if (weight == 0) return 0;

    return cube_index;
}

__device__ __forceinline__
float3 get_node_coordinates(int x, int y, int z, float voxel_scale, Vec3f &volum_origin) {
    float3 position;

    position.x = (x + 0.5f) * voxel_scale + volum_origin(0);
    position.y = (y + 0.5f) * voxel_scale + volum_origin(1);
    position.z = (z + 0.5f) * voxel_scale + volum_origin(2);

    return position;
}

__device__ __forceinline__
float3 vertex_interpolate(float3 &p0, float3 &p1, float f0, float f1) {
    float t = (0.f - f0) / (f1 - f0 + 1e-15f);
    return make_float3(p0.x + t * (p1.x - p0.x),
                       p0.y + t * (p1.y - p0.y),
                       p0.z + t * (p1.z - p0.z));
}


__global__ void get_occupied_voxels_kernel(float2 *tsdf_volume, int3 volume_size,
                                           int *occupied_voxel_indices, int *number_vertices,
                                           int *number_vertices_table, int max_size) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (__all(x >= volume_size.x) || __all(y >= volume_size.y))
        return;

    const int flattened_tid =
            threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = flattened_tid >> 5;
    const int lane_id = lane_ID();

    volatile __shared__ int warps_buffer[32]; // Number of threads / Warp size

    for (int z = 0; z < volume_size.z - 1; ++z) {
        int n_vertices = 0;
        if (x + 1 < volume_size.x && y + 1 < volume_size.y) {
            float tsdf_values[8];
            int cube_index = compute_cube_index(tsdf_volume, volume_size, x, y, z, tsdf_values);
            n_vertices = (cube_index == 0 || cube_index == 255) ? 0 : number_vertices_table[cube_index];
        }

        int total = __popc(__ballot(n_vertices > 0));

        if (total == 0)
            continue;

        if (lane_id == 0) {
            int old = atomicAdd(&global_count, total);
            warps_buffer[warp_id] = old;
        }

        int old_global_voxels_count = warps_buffer[warp_id];

        int offset = binaryExclScan(__ballot(n_vertices > 0));

        if (old_global_voxels_count + offset < max_size && n_vertices > 0) {
            int current_voxel_index = volume_size.y * volume_size.x * z + volume_size.x * y + x;
            occupied_voxel_indices[old_global_voxels_count + offset] = current_voxel_index;
            number_vertices[old_global_voxels_count + offset] = n_vertices;
        }

        bool full = old_global_voxels_count + total >= max_size;

        if (full)
            break;
    }
}

__global__ void generate_triangles_kernel(float2 *tsdf_volume, int3 volume_size, float voxel_scale, Vec3f volume_origin,
                                          int *occupied_voxel_indices, int *vertex_offsets,
                                          int *number_vertices_table, int *triangle_table,
                                          int length, float3 *triangle_buffer) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= length)
        return;

    const int voxel = occupied_voxel_indices[idx];

    const int z = voxel / (volume_size.x * volume_size.y);
    const int y = (voxel - z * volume_size.x * volume_size.y) / volume_size.x;
    const int x = (voxel - z * volume_size.x * volume_size.y) - y * volume_size.x;

    float tsdf_values[8];
    const int cube_index = compute_cube_index(tsdf_volume, volume_size, x, y, z, tsdf_values);

    float3 v[8];
    v[0] = get_node_coordinates(x, y, z, voxel_scale, volume_origin);
    v[1] = get_node_coordinates(x + 1, y, z, voxel_scale, volume_origin);
    v[2] = get_node_coordinates(x + 1, y + 1, z, voxel_scale, volume_origin);
    v[3] = get_node_coordinates(x, y + 1, z, voxel_scale, volume_origin);
    v[4] = get_node_coordinates(x, y, z + 1, voxel_scale, volume_origin);
    v[5] = get_node_coordinates(x + 1, y, z + 1, voxel_scale, volume_origin);
    v[6] = get_node_coordinates(x + 1, y + 1, z + 1, voxel_scale, volume_origin);
    v[7] = get_node_coordinates(x, y + 1, z + 1, voxel_scale, volume_origin);

    float3 vertex_list[12];
    vertex_list[0] = vertex_interpolate(v[0], v[1], tsdf_values[0], tsdf_values[1]);
    vertex_list[1] = vertex_interpolate(v[1], v[2], tsdf_values[1], tsdf_values[2]);
    vertex_list[2] = vertex_interpolate(v[2], v[3], tsdf_values[2], tsdf_values[3]);
    vertex_list[3] = vertex_interpolate(v[3], v[0], tsdf_values[3], tsdf_values[0]);
    vertex_list[4] = vertex_interpolate(v[4], v[5], tsdf_values[4], tsdf_values[5]);
    vertex_list[5] = vertex_interpolate(v[5], v[6], tsdf_values[5], tsdf_values[6]);
    vertex_list[6] = vertex_interpolate(v[6], v[7], tsdf_values[6], tsdf_values[7]);
    vertex_list[7] = vertex_interpolate(v[7], v[4], tsdf_values[7], tsdf_values[4]);
    vertex_list[8] = vertex_interpolate(v[0], v[4], tsdf_values[0], tsdf_values[4]);
    vertex_list[9] = vertex_interpolate(v[1], v[5], tsdf_values[1], tsdf_values[5]);
    vertex_list[10] = vertex_interpolate(v[2], v[6], tsdf_values[2], tsdf_values[6]);
    vertex_list[11] = vertex_interpolate(v[3], v[7], tsdf_values[3], tsdf_values[7]);

    int n_vertices = number_vertices_table[cube_index];

    for (int i = 0; i < n_vertices; i += 3) {
        int index = vertex_offsets[idx] + i;

        int v1 = triangle_table[(cube_index * 16) + i + 0];
        int v2 = triangle_table[(cube_index * 16) + i + 1];
        int v3 = triangle_table[(cube_index * 16) + i + 2];

        triangle_buffer[index + 2] = vertex_list[v1];
        triangle_buffer[index + 1] = vertex_list[v2];
        triangle_buffer[index + 0] = vertex_list[v3];
    }
}

int generate_vertex_offsets(int *number_vertices, int *vertex_offsets, int length) {
    int count = 0;
    for (int i = 0; i < length; ++i) {
        vertex_offsets[i] = count;
        count += number_vertices[i];
    }
    return count;
}

void marching_cubes(float2 *tsdf_volume, int3 &volume_size, float voxel_scale, Vec3f &volume_origin,
                    int *number_vertices_table, int *triangle_table, std::vector<float3> &triangles) {
    int max_size = 2000000;
    int *occupied_voxel_indices;
    CUDA_SAFE_CALL(cudaMalloc((void **) &occupied_voxel_indices, sizeof(int) * max_size));
    int *number_vertices;
    CUDA_SAFE_CALL(cudaMallocManaged((void **) &number_vertices, sizeof(int) * max_size));

    int active_voxels = 0;
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(global_count, &active_voxels, sizeof(int)));

    dim3 block_size(32, 32);
    dim3 grid_size((volume_size.x + block_size.x - 1) / block_size.x,
                   (volume_size.y + block_size.y - 1) / block_size.y);
    get_occupied_voxels_kernel << < grid_size, block_size >> > (tsdf_volume, volume_size, occupied_voxel_indices,
            number_vertices, number_vertices_table, max_size);
    cudaDeviceSynchronize();

    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&active_voxels, global_count, sizeof(int)));

    int *vertex_offsets;
    CUDA_SAFE_CALL(cudaMallocManaged((void **) &vertex_offsets, sizeof(int) * active_voxels));

    int count = generate_vertex_offsets(number_vertices, vertex_offsets, active_voxels);
    float3 *triangle_buffer;
    CUDA_SAFE_CALL(cudaMalloc((void **) &triangle_buffer, sizeof(float3) * count));

    dim3 block(256);
    dim3 grid((active_voxels + block.x - 1) / block.x);
    generate_triangles_kernel << < grid, block >> > (tsdf_volume, volume_size, voxel_scale, volume_origin,
            occupied_voxel_indices, vertex_offsets, number_vertices_table, triangle_table, active_voxels,
            triangle_buffer);
    cudaDeviceSynchronize();

    triangles.resize(count);
    CUDA_SAFE_CALL(cudaMemcpy(triangles.data(), triangle_buffer, sizeof(float3) * count, cudaMemcpyDeviceToHost));

    CUDA_SAFE_FREE(occupied_voxel_indices);
    CUDA_SAFE_FREE(number_vertices);
    CUDA_SAFE_FREE(vertex_offsets);
    CUDA_SAFE_FREE(triangle_buffer);
}