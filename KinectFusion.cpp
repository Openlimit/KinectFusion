#include "KinectFusion.h"
#include "mc_tables.h"

void compute_gradient(float *data, Vec2f *gradient, int width, int height);

void downsample(float *in_data, float *out_data, int out_width, int out_height, int down_factor);

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

void marching_cubes(float2 *tsdf_volume, int3 &volume_size, float voxel_scale, Vec3f &volume_origin,
                    int *number_vertices_table, int *triangle_table, std::vector<float3> &triangles);

void solve(SolverInput &input, SolverState &state, SolverParameters &parameters);

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
        CUDA_SAFE_CALL(cudaMalloc(&depth_pyramid[i], sizeof(float) * size));
        CUDA_SAFE_CALL(cudaMalloc(&cur_vertex_pyramid[i], sizeof(Vec3f) * size));
        CUDA_SAFE_CALL(cudaMalloc(&cur_normal_pyramid[i], sizeof(Vec3f) * size));
        CUDA_SAFE_CALL(cudaMalloc(&pre_vertex_pyramid[i], sizeof(Vec3f) * size));
        CUDA_SAFE_CALL(cudaMalloc(&pre_normal_pyramid[i], sizeof(Vec3f) * size));
    }

    CUDA_SAFE_CALL(cudaMalloc(&depth_map, sizeof(float) * cam_param[0].width * cam_param[0].height));
    CUDA_SAFE_CALL(cudaMalloc(&image_map, sizeof(float) * cam_param[0].width * cam_param[0].height));
    CUDA_SAFE_CALL(cudaMalloc(&tsdfVolume,
                              sizeof(float2) * config.volume_size.x * config.volume_size.y *
                              config.volume_size.z));

    CUDA_SAFE_CALL(cudaMalloc(&triangle_table, sizeof(int) * 256 * 16));
    CUDA_SAFE_CALL(cudaMemcpy(triangle_table, triangle_table_host, sizeof(int) * 256 * 16, cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMalloc(&number_vertices_table, sizeof(int) * 256));
    CUDA_SAFE_CALL(cudaMemcpy(number_vertices_table, number_vertices_table_host, sizeof(int) * 256,
                              cudaMemcpyHostToDevice));

    reset();
}

KinectFusion::~KinectFusion() {
    CUDA_SAFE_FREE(triangle_table);
    CUDA_SAFE_FREE(number_vertices_table);

    CUDA_SAFE_FREE(depth_map);
    CUDA_SAFE_FREE(tsdfVolume);
    for (int i = 0; i < LEVELS; ++i) {
        CUDA_SAFE_FREE(depth_pyramid[i]);
        CUDA_SAFE_FREE(cur_vertex_pyramid[i]);
        CUDA_SAFE_FREE(cur_normal_pyramid[i]);
        CUDA_SAFE_FREE(pre_vertex_pyramid[i]);
        CUDA_SAFE_FREE(pre_normal_pyramid[i]);
    }

    for (int i = 0; i < cachedFrames.size(); ++i) {
        cachedFrames[i].free();
    }
}

void KinectFusion::reset() {
    frame_id = 0;
    cur_pose.setIdentity();
    pose_list.clear();
    CUDA_SAFE_CALL(cudaMemset(tsdfVolume, 0,
                              sizeof(float2) * config.volume_size.x * config.volume_size.y * config.volume_size.z));
}

void KinectFusion::extract_mesh(std::string &path) {
    std::vector<float3> triangles;
    marching_cubes(tsdfVolume, config.volume_size, config.voxel_scale, config.volume_origin, number_vertices_table,
                   triangle_table, triangles);

    std::ofstream file(path);
    for (int i = 0; i < triangles.size(); i += 3) {
        for (int k = 0; k < 3; ++k) {
            float3 point = triangles[i + k];
            file << "v " << point.x << " " << point.y << " " << point.z << std::endl;
        }
        file << "f " << i + 1 << " " << i + 2 << " " << i + 3 << std::endl;
    }
    file.close();
}

void KinectFusion::save_tsdf(std::string &path) {
    int size = config.volume_size.x * config.volume_size.y * config.volume_size.z;

    float2 *tsdfVolume_host = new float2[size];
    CUDA_SAFE_CALL(cudaMemcpy(tsdfVolume_host, tsdfVolume, sizeof(float2) * size, cudaMemcpyDeviceToHost));

    std::ofstream file(path, std::ios::binary);
    for (int i = 0; i < size; ++i) {
        file.write((char *) &tsdfVolume_host[i].x, sizeof(float));
        file.write((char *) &tsdfVolume_host[i].y, sizeof(float));
    }
    file.close();

    delete[] tsdfVolume_host;
}

bool KinectFusion::process(float *depth_frame, float *image_frame) {
    surface_measurement(depth_frame, image_frame);

    cache_data();

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

bool KinectFusion::process(float *depth_frame, float *image_frame, Mat4f &transform){
    surface_measurement(depth_frame, image_frame);

    cache_data();

    cur_pose = transform;
    pose_list.push_back(transform);

    surface_reconstruction();
    frame_id++;
}

void KinectFusion::reFusion() {
    CUDA_SAFE_CALL(cudaMemset(tsdfVolume, 0,
                              sizeof(float2) * config.volume_size.x * config.volume_size.y * config.volume_size.z));
    for (int i = 0; i < pose_list.size(); ++i) {
        update_tsdf(cachedFrames[i].d_depthMap, tsdfVolume, config.volume_origin,
                    cam_param[0].width, cam_param[0].height,
                    config.volume_size, config.voxel_scale,
                    config.truncation_distance, pose_list[i], K[0]);
    }
}

void KinectFusion::optimize() {
    const unsigned int numOfImages = pose_list.size();

    unsigned int nNonLinearIterations = 3;
    unsigned int nLinIterations = 100;
    const int lv = LEVELS - 1;

    //params
    SolverParameters parameters;
    parameters.nNonLinearIterations = nNonLinearIterations;
    parameters.nLinIterations = nLinIterations;
    parameters.denseDistThresh = 5;
    parameters.denseDepthMax = 1000.f;
    parameters.denseDepthMin = 0.f;
    parameters.denseNormalThresh = 0.9;
    parameters.denseColorThresh = 0.1f;
    parameters.denseColorGradientMin = 0.005f;
    parameters.denseOverlapCheckSubsampleFactor = 4;
    parameters.minNumOverlapCorr = 10;
    parameters.minNumDenseCorr = 100;
    parameters.boundingMin = config.volume_origin;
    parameters.boundingMax = config.volume_origin +
                             Vec3f(config.volume_size.x, config.volume_size.y, config.volume_size.z) *
                             config.voxel_scale;

    //input
    SolverInput input;
    input.denseDepthWidth = cam_param[lv].width;
    input.denseDepthHeight = cam_param[lv].height;
    input.numberOfImages = numOfImages;
    input.intrinsics = Vec4f(cam_param[lv].focal_x, cam_param[lv].focal_y,
                             cam_param[lv].principal_x, cam_param[lv].principal_y);
    CUDA_SAFE_CALL(cudaMalloc(&input.d_cacheFrames, sizeof(CUDADataFrame) * numOfImages));
    CUDA_SAFE_CALL(cudaMemcpy(input.d_cacheFrames, cachedFrames.data(), sizeof(CUDADataFrame) * numOfImages,
                              cudaMemcpyHostToDevice));
    input.weightsDenseDepth = new float[nNonLinearIterations];
    input.weightsDenseColor = new float[nNonLinearIterations];
    for (int i = 0; i < nNonLinearIterations; ++i) {
        input.weightsDenseDepth[i] = 1.f;
        input.weightsDenseColor[i] = 1.f;
    }

    // state
    Mat4f *d_transform;
    CUDA_SAFE_CALL(cudaMalloc((void **) &d_transform, sizeof(Mat4f) * numOfImages));
    CUDA_SAFE_CALL(cudaMemcpy(d_transform, pose_list.data(), sizeof(Mat4f) * numOfImages, cudaMemcpyHostToDevice));

    SolverState state;
    CUDA_SAFE_CALL(cudaMalloc(&state.d_xRot, sizeof(Vec3f) * numOfImages));
    CUDA_SAFE_CALL(cudaMalloc(&state.d_xTrans, sizeof(Vec3f) * numOfImages));
    convertMatricesToLiePoses(d_transform, numOfImages, state.d_xRot, state.d_xTrans);

    CUDA_SAFE_CALL(cudaMalloc(&state.d_deltaRot, sizeof(Vec3f) * numOfImages));
    CUDA_SAFE_CALL(cudaMalloc(&state.d_deltaTrans, sizeof(Vec3f) * numOfImages));
    CUDA_SAFE_CALL(cudaMalloc(&state.d_rRot, sizeof(Vec3f) * numOfImages));
    CUDA_SAFE_CALL(cudaMalloc(&state.d_rTrans, sizeof(Vec3f) * numOfImages));
    CUDA_SAFE_CALL(cudaMalloc(&state.d_zRot, sizeof(Vec3f) * numOfImages));
    CUDA_SAFE_CALL(cudaMalloc(&state.d_zTrans, sizeof(Vec3f) * numOfImages));
    CUDA_SAFE_CALL(cudaMalloc(&state.d_pRot, sizeof(Vec3f) * numOfImages));
    CUDA_SAFE_CALL(cudaMalloc(&state.d_pTrans, sizeof(Vec3f) * numOfImages));
    CUDA_SAFE_CALL(cudaMalloc(&state.d_Ap_XRot, sizeof(Vec3f) * numOfImages));
    CUDA_SAFE_CALL(cudaMalloc(&state.d_Ap_XTrans, sizeof(Vec3f) * numOfImages));
    CUDA_SAFE_CALL(cudaMalloc(&state.d_scanAlpha, sizeof(float) * 2));
    CUDA_SAFE_CALL(cudaMalloc(&state.d_rDotzOld, sizeof(float) * numOfImages));
    CUDA_SAFE_CALL(cudaMalloc(&state.d_precondionerRot, sizeof(Vec3f) * numOfImages));
    CUDA_SAFE_CALL(cudaMalloc(&state.d_precondionerTrans, sizeof(Vec3f) * numOfImages));

    CUDA_SAFE_CALL(cudaMalloc(&state.d_denseJtJ, sizeof(float) * 36 * numOfImages * numOfImages));
    CUDA_SAFE_CALL(cudaMalloc(&state.d_denseJtr, sizeof(float) * 6 * numOfImages));

    unsigned int numDenseImPairs = numOfImages * (numOfImages - 1) / 2;
    CUDA_SAFE_CALL(cudaMalloc(&state.d_denseCorrCounts, sizeof(float) * numDenseImPairs));
    CUDA_SAFE_CALL(cudaMalloc(&state.d_denseOverlappingImages, sizeof(uint2) * numDenseImPairs));
    CUDA_SAFE_CALL(cudaMalloc(&state.d_numDenseOverlappingImages, sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc(&state.d_xTransforms, sizeof(Mat4f) * numOfImages));
    CUDA_SAFE_CALL(cudaMalloc(&state.d_xTransformInverses, sizeof(Mat4f) * numOfImages));
#ifdef DEBUG
    CUDA_SAFE_CALL(cudaMalloc(&state.d_sumResidualDEBUG, sizeof(float) * numDenseImPairs));
    CUDA_SAFE_CALL(cudaMalloc(&state.d_numCorrDEBUG, sizeof(int) * numDenseImPairs));
    CUDA_SAFE_CALL(cudaMalloc(&state.d_J, sizeof(float) * 6 * numOfImages));
#endif

    solve(input, state, parameters);
    CUDA_SAFE_CALL(cudaMemcpy(pose_list.data(), state.d_xTransforms,
                              sizeof(Mat4f) * numOfImages, cudaMemcpyDeviceToHost));

    delete[] input.weightsDenseDepth;
    delete[] input.weightsDenseColor;
    CUDA_SAFE_FREE(input.d_cacheFrames);
    CUDA_SAFE_FREE(d_transform);

    CUDA_SAFE_FREE(state.d_xRot);
    CUDA_SAFE_FREE(state.d_xTrans);
    CUDA_SAFE_FREE(state.d_deltaRot);
    CUDA_SAFE_FREE(state.d_deltaTrans);
    CUDA_SAFE_FREE(state.d_rRot);
    CUDA_SAFE_FREE(state.d_rTrans);
    CUDA_SAFE_FREE(state.d_zRot);
    CUDA_SAFE_FREE(state.d_zTrans);
    CUDA_SAFE_FREE(state.d_pRot);
    CUDA_SAFE_FREE(state.d_pTrans);
    CUDA_SAFE_FREE(state.d_Ap_XRot);
    CUDA_SAFE_FREE(state.d_Ap_XTrans);
    CUDA_SAFE_FREE(state.d_scanAlpha);
    CUDA_SAFE_FREE(state.d_rDotzOld);
    CUDA_SAFE_FREE(state.d_precondionerRot);
    CUDA_SAFE_FREE(state.d_precondionerTrans);
    CUDA_SAFE_FREE(state.d_denseJtJ);
    CUDA_SAFE_FREE(state.d_denseJtr);
    CUDA_SAFE_FREE(state.d_denseCorrCounts);
    CUDA_SAFE_FREE(state.d_denseOverlappingImages);
    CUDA_SAFE_FREE(state.d_numDenseOverlappingImages);
    CUDA_SAFE_FREE(state.d_xTransforms);
    CUDA_SAFE_FREE(state.d_xTransformInverses);
#ifdef DEBUG
    CUDA_SAFE_FREE(state.d_sumResidualDEBUG);
    CUDA_SAFE_FREE(state.d_numCorrDEBUG);
    CUDA_SAFE_FREE(state.d_J);
#endif
}

void KinectFusion::cache_data() {
    CUDADataFrame cachedFrame;

    const int lv = LEVELS - 1;
    const int size = cam_param[lv].width * cam_param[lv].height;
    cachedFrame.alloc(cam_param[lv].width, cam_param[lv].height, cam_param[0].width, cam_param[0].height);

    CUDA_SAFE_CALL(cudaMemcpy(cachedFrame.d_depthMap, depth_map,
                              sizeof(float) * cam_param[0].width * cam_param[0].height, cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(cachedFrame.d_depthDownsampled, depth_pyramid[lv],
                              sizeof(float) * size, cudaMemcpyDeviceToDevice));
    copyVec3ToVec4(cachedFrame.d_cameraposDownsampled, cur_vertex_pyramid[lv], size, 1.0f);
    copyVec3ToVec4(cachedFrame.d_normalsDownsampled, cur_normal_pyramid[lv], size, 0.f);

    int factor = cam_param[0].width / cam_param[lv].width;
    downsample(image_map, cachedFrame.d_intensityDownsampled, cam_param[lv].width, cam_param[lv].height, factor);
    compute_gradient(cachedFrame.d_intensityDownsampled, cachedFrame.d_intensityDerivsDownsampled,
                     cam_param[lv].width, cam_param[lv].height);

    cachedFrames.push_back(cachedFrame);
}


void KinectFusion::surface_measurement(float *depth_frame, float *image_frame) {
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

    CUDA_SAFE_CALL(cudaMemcpy(image_map, image_frame, sizeof(float) * cam_param[0].width * cam_param[0].height,
                              cudaMemcpyHostToDevice));
}

bool KinectFusion::pose_estimation() {
    Mat4f pre_pose = cur_pose;

    for (int level = LEVELS - 2; level >= 0; --level) {
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
    for (int level = 0; level < LEVELS - 1; ++level) {
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