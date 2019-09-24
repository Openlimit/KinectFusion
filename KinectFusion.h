//
// Created by meidai on 19-9-16.
//

#ifndef KINECTFUSION_KINECTFUSION_H
#define KINECTFUSION_KINECTFUSION_H

#include "common.h"
#include "cuda/cuda_common.h"

struct CameraParameter {
    int width, height;
    float focal_x, focal_y;
    float principal_x, principal_y;

    CameraParameter() = default;
};

struct KinectConfig {
//    int3 volume_size = make_int3(512, 512, 512);
//    int3 volume_size = make_int3(256, 256, 256);
    int3 volume_size = make_int3(128, 128, 128);
    float voxel_scale = 1.f;
    Vec3f volume_origin = Vec3f(-200.f, -300.f, 200.f);

    float truncation_distance = 25.f;

    int bfilter_kernel_size = 5;
    float bfilter_range_sigma = 5.f;
    float bfilter_spatial_sigma = 5.f;

    std::vector<int> icp_iterations{10, 5, 4};

    float distance_threshold = 10.f;
    float angle_threshold = cosf(3.14159254f * 30.0f / 180.0f);
};

class KinectFusion {
public:
    KinectFusion(CameraParameter &cameraParameter, KinectConfig &config);

    ~KinectFusion();

    bool process(float *depth_frame);

    void reset();

    void extract_mesh(std::string &path);

    void save_tsdf(std::string &path);

    void download_depth_map(std::string &outPath);

    void download_depth_pyramid(std::string &outPath);

    void download_cur_vertex_pyramid(std::string &outPath);

    void download_pre_vertex_pyramid(std::string &outPath);

    void download_cur_vector_and_normal_pyramid(std::string &outPath);

    void download_pre_vector_and_normal_pyramid(std::string &outPath);

    void download_cur_vertex_pyramid_with_pose(std::string &outPath, Mat4f &pose);

    void download_pre_vertex_pyramid_with_pose(std::string &outPath, Mat4f &pose);

//private:
    void surface_measurement(float *depth_frame);

    bool pose_estimation();

    void surface_reconstruction();

    void surface_prediction();

    KinectConfig config;
    int frame_id;

    Mat4f cur_pose;
    std::vector<Mat4f> pose_list;

    CameraParameter cam_param[LEVELS];
    Mat3f K[LEVELS];

    float *depth_pyramid[LEVELS];
    Vec3f *cur_vertex_pyramid[LEVELS];
    Vec3f *cur_normal_pyramid[LEVELS];

    Vec3f *pre_vertex_pyramid[LEVELS];
    Vec3f *pre_normal_pyramid[LEVELS];

    float *depth_map;
    float2 *tsdfVolume;

    int *triangle_table;
    int *number_vertices_table;
};


#endif //KINECTFUSION_KINECTFUSION_H
