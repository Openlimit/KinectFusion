#include "KinectFusion.h"

void bilateral_filter(float *in_data, float *out_data, int width, int height,
                      int kernel_size, float range_sigma, float spatial_sigma);

void compute_vertex_map(float *data, Vec3f *vertex_map, Mat3f &K, int width, int height);

void compute_normal_map(Vec3f *vertex_map, Vec3f *normal_map, int width, int height);

void pyrDown(float *in_data, float *out_data, int out_width, int out_height, int kernel_size, float range_sigma);

void estimate_step(Mat4f &cur_pose, Mat4f &pre_pose, Mat3f &K,
                   Vec3f *cur_vertex_map, Vec3f *cur_normal_map,
                   Vec3f *pre_vertex_map, Vec3f *pre_normal_map,
                   int width, int height, float distance_threshold, float angle_threshold,
                   Mat66d &A, Vec6d &b);

void update_tsdf(float *depth_map, float2 *tsdf_volume, Vec3f &volume_origin,
                 int width, int height,
                 int3 volume_size, float voxel_scale,
                 float truncation_distance, Mat4f &pose, Mat3f &K);

void raycast_tsdf(float2 *tsdf_volume, Vec3f *vertex_map, Vec3f *normal_map,
                  int width, int height, float truncation_distance,
                  int3 &volume_size, float voxel_scale, Vec3f &volume_origin,
                  Mat3f &K, Mat4f &pose);

KinectFusion::KinectFusion(CameraParameter &cameraParameter, KinectConfig &kinectConfig) {
    cam_param[0] = cameraParameter;
    config = kinectConfig;

    for (int i = 1; i < LEVELS; ++i) {
        cam_param[i].width = cam_param[i - 1].width / 2;
        cam_param[i].height = cam_param[i - 1].height / 2;
        cam_param[i].focal_x = cam_param[i - 1].focal_x / 2;
        cam_param[i].focal_y = cam_param[i - 1].focal_y / 2;
        cam_param[i].principal_x = cam_param[i - 1].principal_x / 2;
        cam_param[i].principal_y = cam_param[i - 1].principal_y / 2;
    }

    for (int i = 0; i < LEVELS; ++i) {
        K[i].setZero();
        K[i](0, 0) = cam_param[i].focal_x;
        K[i](0, 2) = cam_param[i].principal_x;
        K[i](1, 1) = cam_param[i].focal_y;
        K[i](1, 2) = cam_param[i].principal_y;
        K[i](2, 2) = 1;
    }

    for (int i = 0; i < LEVELS; ++i) {
        int size = cam_param[i].width * cam_param[i].height;
        CUDA_SAFE_CALL(cudaMalloc((void **) &depth_pyramid[i], sizeof(float) * size));
        CUDA_SAFE_CALL(cudaMalloc((void **) &cur_vertex_pyramid[i], sizeof(Vec3f) * size));
        CUDA_SAFE_CALL(cudaMalloc((void **) &cur_normal_pyramid[i], sizeof(Vec3f) * size));
        CUDA_SAFE_CALL(cudaMalloc((void **) &pre_vertex_pyramid[i], sizeof(Vec3f) * size));
        CUDA_SAFE_CALL(cudaMalloc((void **) &pre_normal_pyramid[i], sizeof(Vec3f) * size));
    }

    CUDA_SAFE_CALL(cudaMalloc((void **) &depth_map, sizeof(float) * cam_param[0].width * cam_param[0].height));
    CUDA_SAFE_CALL(cudaMalloc((void **) &tsdfVolume,
                              sizeof(float2) * config.volume_size.x * config.volume_size.y *
                              config.volume_size.z));

    reset();
}

KinectFusion::~KinectFusion() {
    CUDA_SAFE_FREE(depth_map);
    CUDA_SAFE_FREE(tsdfVolume);
    for (int i = 0; i < LEVELS; ++i) {
        CUDA_SAFE_FREE(depth_pyramid[i]);
        CUDA_SAFE_FREE(cur_vertex_pyramid[i]);
        CUDA_SAFE_FREE(cur_normal_pyramid[i]);
        CUDA_SAFE_FREE(pre_vertex_pyramid[i]);
        CUDA_SAFE_FREE(pre_normal_pyramid[i]);
    }
}

void KinectFusion::reset() {
    frame_id = 0;
    cur_pose.setIdentity();
    pose_list.clear();
    CUDA_SAFE_CALL(cudaMemset(tsdfVolume, 0,
                              sizeof(float2) * config.volume_size.x * config.volume_size.y * config.volume_size.z));
}

bool KinectFusion::process(float *depth_frame) {
    surface_measurement(depth_frame);

    bool icp_success = true;
    if (frame_id > 0) {
        icp_success = pose_estimation();
    }
    if (!icp_success) {
        std::cout << "icp fail" << std::endl;
        return false;
    }

    pose_list.push_back(cur_pose);

    surface_reconstruction();

    surface_prediction();

    frame_id++;
}


void KinectFusion::surface_measurement(float *depth_frame) {
    CUDA_SAFE_CALL(cudaMemcpy(depth_map, depth_frame, sizeof(float) * cam_param[0].width * cam_param[0].height,
                              cudaMemcpyHostToDevice));

    bilateral_filter(depth_map, depth_pyramid[0], cam_param[0].width, cam_param[0].height,
                     config.bfilter_kernel_size, config.bfilter_range_sigma, config.bfilter_spatial_sigma);

    for (int i = 1; i < LEVELS; ++i) {
        pyrDown(depth_pyramid[i - 1], depth_pyramid[i], cam_param[i].width, cam_param[i].height,
                config.bfilter_kernel_size, config.bfilter_range_sigma);
    }

    for (int i = 0; i < LEVELS; ++i) {
        compute_vertex_map(depth_pyramid[i], cur_vertex_pyramid[i], K[i], cam_param[i].width, cam_param[i].height);
        compute_normal_map(cur_vertex_pyramid[i], cur_normal_pyramid[i], cam_param[i].width, cam_param[i].height);
    }
}

bool KinectFusion::pose_estimation() {
    Mat4f pre_pose = cur_pose;

    for (int level = LEVELS - 1; level >= 0; --level) {
        for (int iter = 0; iter < config.icp_iterations[level]; ++iter) {
            Mat66d A;
            Vec6d b;

            estimate_step(cur_pose, pre_pose, K[level],
                          cur_vertex_pyramid[level], cur_normal_pyramid[level],
                          pre_vertex_pyramid[level], pre_normal_pyramid[level],
                          cam_param[level].width, cam_param[level].height,
                          config.distance_threshold, config.angle_threshold, A, b);

            double det = A.determinant();
            if (fabs(det) < 100000 /*1e-15*/ || std::isnan(det))
                return false;
            Vec6f result = A.fullPivLu().solve(b).cast<float>();
//            std::cout << result << std::endl << std::endl;

            // Update pose
            float alpha = result(0);
            float beta = result(1);
            float gamma = result(2);
            Eigen::AngleAxisf rz(gamma, Eigen::Vector3f::UnitZ());
            Eigen::AngleAxisf ry(beta, Eigen::Vector3f::UnitY());
            Eigen::AngleAxisf rx(alpha, Eigen::Vector3f::UnitX());
            Mat3f rotation(rz * ry * rx);

            Mat4f inc;
            inc.setIdentity();
            inc.block(0, 0, 3, 3) = rotation;
            inc.block(0, 3, 3, 1) = result.tail(3);
            cur_pose = inc * cur_pose;
        }
    }

    return true;
}

void KinectFusion::surface_reconstruction() {
    update_tsdf(depth_map, tsdfVolume, config.volume_origin, cam_param[0].width, cam_param[0].height,
                config.volume_size, config.voxel_scale,
                config.truncation_distance, cur_pose, K[0]);
}

void KinectFusion::surface_prediction() {
    for (int level = 0; level < LEVELS; ++level) {
        raycast_tsdf(tsdfVolume, pre_vertex_pyramid[level], pre_normal_pyramid[level],
                     cam_param[level].width, cam_param[level].height,
                     config.truncation_distance, config.volume_size, config.voxel_scale, config.volume_origin,
                     K[level], cur_pose);
    }
}

void KinectFusion::download_depth_map(std::string &outPath) {
    float *depth = new float[cam_param[0].width * cam_param[0].height];
    CUDA_SAFE_CALL(cudaMemcpy(depth, depth_map,
                              sizeof(float) * cam_param[0].width * cam_param[0].height,
                              cudaMemcpyDeviceToHost));

    std::string path = outPath + "depth_map_" + std::to_string(frame_id) + ".xyz";
    std::cout << path << std::endl;
    std::ofstream file(path.c_str());
    for (int i = 0; i < cam_param[0].height; ++i) {
        for (int j = 0; j < cam_param[0].width; ++j) {
            if (depth[i * cam_param[0].width + j] == 0.f)
                continue;
            Vec3f p(j, i, 1);
            Vec3f point = depth[i * cam_param[0].width + j] * K[0].inverse() * p;
            file << point(0) << " " << point(1) << " " << point(2) << std::endl;
        }
    }
    file.close();
    delete[] depth;
}

void KinectFusion::download_depth_pyramid(std::string &outPath) {
    for (int level = 0; level < LEVELS; ++level) {
        float *depth = new float[cam_param[level].width * cam_param[level].height];
        CUDA_SAFE_CALL(cudaMemcpy(depth, depth_pyramid[level],
                                  sizeof(float) * cam_param[level].width * cam_param[level].height,
                                  cudaMemcpyDeviceToHost));

        std::string path = outPath + "depth_" + std::to_string(level) + ".xyz";
        std::cout << path << std::endl;
        std::ofstream file(path.c_str());
        for (int i = 0; i < cam_param[level].height; ++i) {
            for (int j = 0; j < cam_param[level].width; ++j) {
                if (depth[i * cam_param[level].width + j] == 0.f)
                    continue;
                Vec3f p(j, i, 1);
                Vec3f point = depth[i * cam_param[level].width + j] * K[level].inverse() * p;
                file << point(0) << " " << point(1) << " " << point(2) << std::endl;
            }
        }
        file.close();
        delete[] depth;
    }
}

void KinectFusion::download_cur_vertex_pyramid(std::string &outPath) {
    for (int level = 0; level < LEVELS; ++level) {
        Vec3f *vertex_map = new Vec3f[cam_param[level].width * cam_param[level].height];
        CUDA_SAFE_CALL(cudaMemcpy(vertex_map, cur_vertex_pyramid[level],
                                  sizeof(Vec3f) * cam_param[level].width * cam_param[level].height,
                                  cudaMemcpyDeviceToHost));

        std::string path = outPath + "cur_vertex_" + std::to_string(level) + "_" + std::to_string(frame_id) + +".xyz";
        std::cout << path << std::endl;
        std::ofstream file(path.c_str());
        for (int i = 0; i < cam_param[level].height; ++i) {
            for (int j = 0; j < cam_param[level].width; ++j) {
                Vec3f point = vertex_map[i * cam_param[level].width + j];
                if (point(0) == 0.f && point(1) == 0.f && point(2) == 0.f)
                    continue;
                file << point(0) << " " << point(1) << " " << point(2) << std::endl;
            }
        }
        file.close();
        delete[] vertex_map;
    }
}

void KinectFusion::download_cur_vertex_pyramid_with_pose(std::string &outPath, Mat4f &pose) {
    for (int level = 0; level < 1; ++level) {
        Vec3f *vertex_map = new Vec3f[cam_param[level].width * cam_param[level].height];
        CUDA_SAFE_CALL(cudaMemcpy(vertex_map, cur_vertex_pyramid[level],
                                  sizeof(Vec3f) * cam_param[level].width * cam_param[level].height,
                                  cudaMemcpyDeviceToHost));

        std::string path =
                outPath + "pose_cur_vertex_" + std::to_string(level) + "_" + std::to_string(frame_id) + +".xyz";
        std::cout << path << std::endl;
        std::ofstream file(path.c_str());
        for (int i = 0; i < cam_param[level].height; ++i) {
            for (int j = 0; j < cam_param[level].width; ++j) {
                Vec3f point = vertex_map[i * cam_param[level].width + j];
                if (point(0) == 0.f && point(1) == 0.f && point(2) == 0.f)
                    continue;
                Vec4f pp(point(0), point(1), point(2), 1);
                pp = pose * pp;
                file << pp(0) << " " << pp(1) << " " << pp(2) << std::endl;
            }
        }
        file.close();
        delete[] vertex_map;
    }
}

void KinectFusion::download_pre_vertex_pyramid_with_pose(std::string &outPath, Mat4f &pose) {
    for (int level = 0; level < 1; ++level) {
        Vec3f *vertex_map = new Vec3f[cam_param[level].width * cam_param[level].height];
        CUDA_SAFE_CALL(cudaMemcpy(vertex_map, pre_vertex_pyramid[level],
                                  sizeof(Vec3f) * cam_param[level].width * cam_param[level].height,
                                  cudaMemcpyDeviceToHost));

        std::string path =
                outPath + "pose_pre_vertex_" + std::to_string(level) + "_" + std::to_string(frame_id) + +".xyz";
        std::cout << path << std::endl;
        std::ofstream file(path.c_str());
        for (int i = 0; i < cam_param[level].height; ++i) {
            for (int j = 0; j < cam_param[level].width; ++j) {
                Vec3f point = vertex_map[i * cam_param[level].width + j];
                if (point(0) == 0.f && point(1) == 0.f && point(2) == 0.f)
                    continue;
                Vec4f pp(point(0), point(1), point(2), 1);
                pp = pose * pp;
                file << pp(0) << " " << pp(1) << " " << pp(2) << std::endl;
            }
        }
        file.close();
        delete[] vertex_map;
    }
}

void KinectFusion::download_cur_vector_and_normal_pyramid(std::string &outPath) {
    for (int level = 0; level < LEVELS; ++level) {
        Vec3f *vertex_map = new Vec3f[cam_param[level].width * cam_param[level].height];
        CUDA_SAFE_CALL(cudaMemcpy(vertex_map, cur_vertex_pyramid[level],
                                  sizeof(Vec3f) * cam_param[level].width * cam_param[level].height,
                                  cudaMemcpyDeviceToHost));
        Vec3f *normal_map = new Vec3f[cam_param[level].width * cam_param[level].height];
        CUDA_SAFE_CALL(cudaMemcpy(normal_map, cur_normal_pyramid[level],
                                  sizeof(Vec3f) * cam_param[level].width * cam_param[level].height,
                                  cudaMemcpyDeviceToHost));

        std::string path = outPath + "cur_vn_" + std::to_string(level) + "_" + std::to_string(frame_id) + ".xyz";
        std::cout << path << std::endl;
        std::ofstream file(path.c_str());
        for (int i = 0; i < cam_param[level].height; ++i) {
            for (int j = 0; j < cam_param[level].width; ++j) {
                Vec3f point = vertex_map[i * cam_param[level].width + j];
                Vec3f normal = normal_map[i * cam_param[level].width + j];
                if (point(0) == 0.f && point(1) == 0.f && point(2) == 0.f)
                    continue;
                if (normal(0) == 0.f && normal(1) == 0.f && normal(2) == 0.f)
                    continue;
//                file << point(0) << " " << point(1) << " " << point(2) << " "
//                     << normal(0) << " " << normal(1) << " " << normal(2) << std::endl;
                point += normal * 5;
                file << point(0) << " " << point(1) << " " << point(2) << std::endl;
            }
        }
        file.close();
        delete[] vertex_map;
        delete[] normal_map;
    }
}

void KinectFusion::download_pre_vertex_pyramid(std::string &outPath) {
    for (int level = 0; level < 1; ++level) {
        Vec3f *vertex_map = new Vec3f[cam_param[level].width * cam_param[level].height];
        CUDA_SAFE_CALL(cudaMemcpy(vertex_map, pre_vertex_pyramid[level],
                                  sizeof(Vec3f) * cam_param[level].width * cam_param[level].height,
                                  cudaMemcpyDeviceToHost));

        std::string path = outPath + "pre_vertex_" + std::to_string(level) + "_" + std::to_string(frame_id) + ".xyz";
        std::cout << path << std::endl;
        std::ofstream file(path.c_str());
        for (int i = 0; i < cam_param[level].height; ++i) {
            for (int j = 0; j < cam_param[level].width; ++j) {
                Vec3f point = vertex_map[i * cam_param[level].width + j];
                if (point(0) == 0.f && point(1) == 0.f && point(2) == 0.f)
                    continue;
                file << point(0) << " " << point(1) << " " << point(2) << std::endl;
            }
        }
        file.close();
        delete[] vertex_map;
    }
}

void KinectFusion::download_pre_vector_and_normal_pyramid(std::string &outPath) {
    for (int level = 0; level < LEVELS; ++level) {
        Vec3f *vertex_map = new Vec3f[cam_param[level].width * cam_param[level].height];
        CUDA_SAFE_CALL(cudaMemcpy(vertex_map, pre_vertex_pyramid[level],
                                  sizeof(Vec3f) * cam_param[level].width * cam_param[level].height,
                                  cudaMemcpyDeviceToHost));
        Vec3f *normal_map = new Vec3f[cam_param[level].width * cam_param[level].height];
        CUDA_SAFE_CALL(cudaMemcpy(normal_map, pre_normal_pyramid[level],
                                  sizeof(Vec3f) * cam_param[level].width * cam_param[level].height,
                                  cudaMemcpyDeviceToHost));

        std::string path = outPath + "pre_vn_" + std::to_string(level) + "_" + std::to_string(frame_id) + ".xyz";
        std::cout << path << std::endl;
        std::ofstream file(path.c_str());
        for (int i = 0; i < cam_param[level].height; ++i) {
            for (int j = 0; j < cam_param[level].width; ++j) {
                Vec3f point = vertex_map[i * cam_param[level].width + j];
                Vec3f normal = normal_map[i * cam_param[level].width + j];
                if (point(0) == 0.f && point(1) == 0.f && point(2) == 0.f)
                    continue;
                if (normal(0) == 0.f && normal(1) == 0.f && normal(2) == 0.f)
                    continue;
//                file << point(0) << " " << point(1) << " " << point(2) << " "
//                     << normal(0) << " " << normal(1) << " " << normal(2) << std::endl;
                point += normal * 5;
                file << point(0) << " " << point(1) << " " << point(2) << std::endl;
            }
        }
        file.close();
        delete[] vertex_map;
        delete[] normal_map;
    }
}

void KinectFusion::pose_test(float *depth1, float *depth2) {
    CUDA_SAFE_CALL(cudaMemcpy(depth_map, depth2, sizeof(float) * cam_param[0].width * cam_param[0].height,
                              cudaMemcpyHostToDevice));

    bilateral_filter(depth_map, depth_pyramid[0], cam_param[0].width, cam_param[0].height,
                     config.bfilter_kernel_size, config.bfilter_range_sigma, config.bfilter_spatial_sigma);

    for (int i = 1; i < LEVELS; ++i) {
        pyrDown(depth_pyramid[i - 1], depth_pyramid[i], cam_param[i].width, cam_param[i].height,
                config.bfilter_kernel_size, config.bfilter_range_sigma);
    }

    for (int i = 0; i < LEVELS; ++i) {
        compute_vertex_map(depth_pyramid[i], cur_vertex_pyramid[i], K[i], cam_param[i].width, cam_param[i].height);
        compute_normal_map(cur_vertex_pyramid[i], cur_normal_pyramid[i], cam_param[i].width, cam_param[i].height);
    }

    CUDA_SAFE_CALL(cudaMemcpy(depth_pyramid[0], depth1, sizeof(float) * cam_param[0].width * cam_param[0].height,
                              cudaMemcpyHostToDevice));

//    bilateral_filter(depth_map, depth_pyramid[0], cam_param[0].width, cam_param[0].height,
//                     config.bfilter_kernel_size, config.bfilter_range_sigma, config.bfilter_spatial_sigma);

    for (int i = 1; i < LEVELS; ++i) {
        pyrDown(depth_pyramid[i - 1], depth_pyramid[i], cam_param[i].width, cam_param[i].height,
                config.bfilter_kernel_size, config.bfilter_range_sigma);
    }

    Vec3f min_point = config.volume_origin;
    Vec3f max_point(config.volume_size.x * config.voxel_scale + min_point(0),
                    config.volume_size.y * config.voxel_scale + min_point(1),
                    config.volume_size.z * config.voxel_scale + min_point(2));

    for (int i = 0; i < LEVELS; ++i) {
        compute_vertex_map_cut(depth_pyramid[i], pre_vertex_pyramid[i], K[i], cam_param[i].width, cam_param[i].height,
                               max_point, min_point);
//        compute_vertex_map(depth_pyramid[i], pre_vertex_pyramid[i], K[i], cam_param[i].width, cam_param[i].height);
        compute_normal_map(pre_vertex_pyramid[i], pre_normal_pyramid[i], cam_param[i].width, cam_param[i].height);
    }

    Mat4f cur = Mat4f::Identity();
    float alpha = 0.001;
    float beta = 0.001;
    float gamma = 0.001;
    Eigen::AngleAxisf rz(gamma, Eigen::Vector3f::UnitZ());
    Eigen::AngleAxisf ry(beta, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf rx(alpha, Eigen::Vector3f::UnitX());
    Mat3f rotation(rz * ry * rx);
    cur.block(0, 0, 3, 3) = rotation;
    cur.block(0, 3, 3, 1) = Vec3f(0.2, 1.2, -2.2f);

    Mat4f pre = cur;

    for (int level = LEVELS - 1; level >= 0; --level) {
        for (int iter = 0; iter < config.icp_iterations[level]; ++iter) {
            Mat66d A;
            Vec6d b;

            estimate_step(cur, pre, K[level],
                          cur_vertex_pyramid[level], cur_normal_pyramid[level],
                          pre_vertex_pyramid[level], pre_normal_pyramid[level],
                          cam_param[level].width, cam_param[level].height,
                          config.distance_threshold, config.angle_threshold, A, b);

            double det = A.determinant();
            if (fabs(det) < 100000 /*1e-15*/ || std::isnan(det)) {
                std::cout << "icp fail" << std::endl;
                break;
            }
            Vec6f result = A.fullPivLu().solve(b).cast<float>();
//            std::cout << result << std::endl << std::endl;

            // Update pose
            float alpha = result(0);
            float beta = result(1);
            float gamma = result(2);
            Eigen::AngleAxisf rz(gamma, Eigen::Vector3f::UnitZ());
            Eigen::AngleAxisf ry(beta, Eigen::Vector3f::UnitY());
            Eigen::AngleAxisf rx(alpha, Eigen::Vector3f::UnitX());
            Mat3f rotation(rz * ry * rx);

            Mat4f inc;
            inc.setIdentity();
            inc.block(0, 0, 3, 3) = rotation;
            inc.block(0, 3, 3, 1) = result.tail(3);
            cur = inc * cur;
        }
    }

    std::string out_path = "/home/meidai/下载/kinectfusion/";
    Mat4f I = Mat4f::Identity();
    download_cur_vertex_pyramid_with_pose(out_path, cur);
    download_pre_vertex_pyramid_with_pose(out_path, I);
}