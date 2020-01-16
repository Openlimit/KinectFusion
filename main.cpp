#include "KinectFusion.h"
#include <dirent.h>

struct Frustum {
    Vec4f points[8];
};

void make_default_origin_frustum(Frustum &origin_frustum,
                                 double far_half_width, double far_half_height,
                                 double near_half_width, double near_half_height,
                                 double z_distance) {
    origin_frustum.points[0] = Vec4f(-near_half_width, near_half_height, 0, 1);
    origin_frustum.points[1] = Vec4f(-near_half_width, -near_half_height, 0, 1);
    origin_frustum.points[2] = Vec4f(near_half_width, near_half_height, 0, 1);
    origin_frustum.points[3] = Vec4f(near_half_width, -near_half_height, 0, 1);
    origin_frustum.points[4] = Vec4f(-far_half_width, far_half_height, -z_distance, 1);
    origin_frustum.points[5] = Vec4f(-far_half_width, -far_half_height, -z_distance, 1);
    origin_frustum.points[6] = Vec4f(far_half_width, far_half_height, -z_distance, 1);
    origin_frustum.points[7] = Vec4f(far_half_width, -far_half_height, -z_distance, 1);
}

void save_camera_trajectory(std::string &filepath, std::vector<Mat4f> &transforms) {
    std::vector<Frustum> frustum_vec;

    Frustum global_frustum;
    make_default_origin_frustum(global_frustum, 5, 5, 10, 10, 10);
    for (int i = 0; i < transforms.size(); ++i) {
        Frustum trans_frustum;
        Mat4f g2f = transforms[i].inverse();
        for (int j = 0; j < 8; j++) {
            trans_frustum.points[j] = g2f * global_frustum.points[j];
        }
        frustum_vec.push_back(trans_frustum);
    }

    std::ofstream out(filepath.c_str());
    for (int i = 0; i < frustum_vec.size(); i++) {
        for (int j = 0; j < 8; j++) {
            out << "v " << frustum_vec[i].points[j](0) << " " << frustum_vec[i].points[j](1)
                << " " << frustum_vec[i].points[j](2) << std::endl;
        }
    }
    for (int i = 0; i < frustum_vec.size(); i++) {
        int gap = i * 8;
        out << "f " << (gap + 1) << " " << (gap + 2) << " " << (gap + 4) << " " << (gap + 3) << std::endl;
        out << "f " << (gap + 5) << " " << (gap + 7) << " " << (gap + 8) << " " << (gap + 6) << std::endl;
        out << "f " << (gap + 1) << " " << (gap + 5) << " " << (gap + 6) << " " << (gap + 2) << std::endl;
        out << "f " << (gap + 3) << " " << (gap + 4) << " " << (gap + 8) << " " << (gap + 7) << std::endl;
        out << "f " << (gap + 1) << " " << (gap + 3) << " " << (gap + 7) << " " << (gap + 5) << std::endl;
        out << "f " << (gap + 2) << " " << (gap + 6) << " " << (gap + 8) << " " << (gap + 4) << std::endl;
    }
    out.close();
}

void saveMat(std::string &path, Mat4f &mat) {
    std::ofstream file(path.c_str(), std::ios::binary);
    float tmp[16];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            tmp[i * 4 + j] = mat(i, j);
        }
    }
    file.write((char *) tmp, sizeof(float) * 16);
    file.close();
}

float float16ToNativeFloat(uint16_t value) {
    union FP32 {
        uint32_t u;
        float f;
    };

    const union FP32 magic = {(254UL - 15UL) << 23};
    const union FP32 was_inf_nan = {(127UL + 16UL) << 23};
    union FP32 out;

    out.u = (value & 0x7FFFU) << 13;
    out.f *= magic.f;
    if (out.f >= was_inf_nan.f) {
        out.u |= 255UL << 23;
    }
    out.u |= (value & 0x8000UL) << 16;

    return out.f;
}

void readDepthFloat16(std::string &name, float *depth, CameraParameter &param) {
    std::ifstream fs(name.c_str(), std::ios::binary);
    for (int i = 0; i < param.height * param.width; ++i) {
        float tmpf;
        uint16_t tmp;
        fs.read((char *) &tmp, sizeof(uint16_t));
        tmpf = float16ToNativeFloat(tmp);

        int x = i % param.width;
        int y = i / param.width;
        int idx = (param.height - y - 1) * param.width + x;
        if (tmpf != tmpf) {
            depth[idx] = 0;
            continue;
        }
        depth[idx] = 1000.0f / tmpf;
        if (depth[idx] > 1000)
            depth[idx] = 0;
    }
    fs.close();
}

void readDepth(std::string &name, float *depth, CameraParameter &param) {
    std::ifstream fs(name.c_str(), std::ios::binary);
    for (int i = 0; i < param.height * param.width; ++i) {
        float tmpf;
        fs.read((char *) &tmpf, sizeof(float));

        int x = i % param.width;
        int y = i / param.width;
        int idx = (param.height - y - 1) * param.width + x;
        if (tmpf != tmpf) {
            depth[idx] = 0;
            continue;
        }
        depth[idx] = tmpf;
        if (depth[idx] > 1000)
            depth[idx] = 0;
    }
    fs.close();
}

void readImage(std::string &name, float *imageIntensity, CameraParameter &param) {
    std::ifstream fs(name.c_str(), std::ios::binary);
    for (int i = 0; i < param.height * param.width * 4; ++i) {
        char b, g, r, a;
        fs.read(&b, sizeof(char));
        fs.read(&g, sizeof(char));
        fs.read(&r, sizeof(char));
        fs.read(&a, sizeof(char));

        ////图片分辨率是depth两倍
        int org_x = i % (param.width * 2);
        int org_y = i / (param.width * 2);
        if (org_x % 2 == 0 && org_y % 2 == 0) {
            float intensity = (0.299f * r + 0.587f * g + 0.114f * b) / 255.0f;
            int x = org_x / 2;
            int y = org_y / 2;
            int idx = (param.height - y - 1) * param.width + x;
            imageIntensity[idx] = intensity;
        }
    }
    fs.close();
}

void readFiles(std::vector<std::string> &file_list, std::string &path) {
    struct dirent *ptr;
    DIR *dir = opendir(path.c_str());
    while ((ptr = readdir(dir)) != NULL) {
        //跳过'.'和'..'两个目录
        if (ptr->d_name[0] == '.')
            continue;
        file_list.push_back(ptr->d_name);
    }
    closedir(dir);
}

bool compare(const std::string &a, const std::string &b) {
    int m = atoi(a.c_str());
    int n = atoi(b.c_str());
    return m < n;
}

bool compare_realsense(const std::string &a, const std::string &b) {
    int m = atoi(a.substr(10).c_str());
    int n = atoi(b.substr(10).c_str());
    return m < n;
}

void read3DMarks(std::string &name, std::vector<Vec3f> &marks) {
    std::ifstream fs(name.c_str(), std::ios::binary);
    float tmp[3];
    while (!fs.eof()) {
        fs.read((char *) tmp, sizeof(float) * 3);
        Vec3f p(tmp[0], tmp[1], tmp[2]);
        marks.push_back(p);
    }
    fs.close();
}

void save3DMarks(std::string &name, std::vector<Vec3f> &marks) {
    std::ofstream fs(name.c_str());
    for (int i = 0; i < marks.size(); ++i) {
        fs << marks[i](0) << " " << marks[i](1) << " " << marks[i](2) << std::endl;
    }
    fs.close();
}


void readMarks(std::string &name, std::vector<Vec2f> &marks, int video_depth_time = 1) {
    std::ifstream fs(name.c_str(), std::ios::binary);
    float tmp[150];
    fs.read((char *) tmp, sizeof(float) * 150);
    for (int i = 0; i < 75; ++i) {
        Vec2f p(tmp[i * 2] / video_depth_time, tmp[i * 2 + 1] / video_depth_time);
        marks.push_back(p);
    }
    fs.close();
}

void project_inv(Vec3f &point, Vec3f &p, CameraParameter &param) {
    p(0) = (point(0) - param.principal_x) * point(2) / param.focal_x;
    p(1) = (point(1) - param.principal_y) * point(2) / param.focal_y;
    p(2) = point(2);
}

void compute3DMarks(std::string &depth_path, std::string &marks_path, std::vector<Vec3f> &marks3D,
                    CameraParameter &param, bool isFloat16 = true, int video_depth_time = 1) {
    float *depth = new float[param.height * param.width];
    if (isFloat16)
        readDepthFloat16(depth_path, depth, param);
    else
        readDepth(depth_path, depth, param);

    std::vector<Vec2f> marks2D;
    readMarks(marks_path, marks2D, video_depth_time);

    for (int i = 15; i < marks2D.size(); ++i) {
        marks2D[i](1) = param.height - 1 - marks2D[i](1);
        int idx = int(marks2D[i](1)) * param.width + int(marks2D[i](0));
        if (depth[idx] > 0.f) {
            Vec3f point(marks2D[i](0), marks2D[i](1), depth[idx]);
            Vec3f p;
            project_inv(point, p, param);
            marks3D.push_back(p);
        }
    }

    delete[] depth;
}

void readTrans(std::string &name, Mat4f &trans) {
    std::ifstream fs(name.c_str(), std::ios::binary);
    float tmp[16];
    fs.read((char *) tmp, sizeof(float) * 16);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            trans(i, j) = tmp[i * 4 + j];
        }
    }
    fs.close();

    Mat4f I;
    I.setIdentity();
    I(2, 2) = -1;
    trans = I * trans * I;
}

//int main() {
//    std::string dir = "/home/meidai/下载/rotation_head/840412061044/";
//    std::string path = dir + "depth/";
//    std::string image_path = dir + "bgra/";
//    std::string out_dir = "/home/meidai/下载/kinectfusion/";
//    std::string trans_dir = dir + "trans/";
//    std::string trans_op_dir = dir + "trans_op/";
//    std::string first_depth_path = dir + "depth/d415_depth0.bin";
//    std::string marks_path = dir + "landmarks2d.bin";
//
//    CameraParameter param;
//    param.width = 640;
//    param.height = 480;
//    float focal = 596.683;
//    param.focal_x = focal;
//    param.focal_y = focal;
//    param.principal_x = 324.212;
//    param.principal_y = 234.038;
//
//    std::vector<Vec3f> marks;
//    compute3DMarks(first_depth_path, marks_path, marks, param, false, 1);
//
//    Vec3f max_marks(-1e9, -1e9, -1e9), min_marks(1e9, 1e9, 1e9);
//    for (int i = 0; i < marks.size(); ++i) {
//        for (int j = 0; j < 3; ++j) {
//            if (marks[i](j) < min_marks(j))
//                min_marks(j) = marks[i](j);
//            if (marks[i](j) > max_marks(j))
//                max_marks(j) = marks[i](j);
//        }
//    }
//    float faceRatio = 2.2;
//    float fw = (max_marks(1) - min_marks(1)) * faceRatio;
//    float half_fw = fw * 0.5f;
//    float xCenter = 0.5f * (max_marks(0) + min_marks(0));
//    float yCenter = 0.8f * max_marks(1) + 0.2f * min_marks(1);
//    min_marks(0) = xCenter - half_fw;
//    min_marks(1) = yCenter - half_fw;
//    min_marks(2) -= 15;
//
//    KinectConfig config;
//    config.voxel_scale = fw / (config.volume_size.x - 1);
//    config.truncation_distance = config.voxel_scale * 10;
//    config.volume_origin = min_marks;
//
//    KinectFusion kinectFusion(param, config);
//
//    std::vector<std::string> file_list;
//    readFiles(file_list, path);
//    std::sort(file_list.begin(), file_list.end(), compare_realsense);
//
//    time_t t00 = time(0);
//    float *depth = new float[param.height * param.width];
//    float *image = new float[param.height * param.width];
//    for (int i = 0; i < file_list.size(); ++i) {
//        std::string name = path + file_list[i];
//        std::string image_name = image_path + file_list[i];
//        std::cout << name << std::endl;
//        readDepth(name, depth, param);
////        readImage(image_name, image, param);
//        kinectFusion.process(depth, image);
//    }
//    time_t t01 = time(0);
//    printf("process time: %ld\n", t01 - t00);
//
//    std::string save_path = out_dir + "trajectory_org.obj";
//    save_camera_trajectory(save_path, kinectFusion.pose_list);
//    std::string mesh_path = out_dir + "mesh128_org.bin";
//    kinectFusion.extract_mesh_bin(mesh_path);
//    for (int i = 0; i < file_list.size(); i++) {
//        std::string trans_path = trans_dir + file_list[i];
//        saveMat(trans_path, kinectFusion.pose_list[i]);
//    }
//
////    time_t t1 = time(0);
////    kinectFusion.optimize();
////    time_t t2 = time(0);
////    printf("optimize time: %ld\n", t2 - t1);
////
////    kinectFusion.reFusion();
////    time_t t3 = time(0);
////    printf("reFusion time: %ld\n", t3 - t2);
////
////    std::string save_path_op = out_dir + "trajectory_op.obj";
////    save_camera_trajectory(save_path_op, kinectFusion.pose_list);
////    std::string mesh_path_op = out_dir + "mesh128_op.bin";
////    kinectFusion.extract_mesh_bin(mesh_path_op);
////    for (int i = 0; i < file_list.size(); i++) {
////        std::string trans_path = trans_op_dir + file_list[i];
////        saveMat(trans_path, kinectFusion.pose_list[i]);
////    }
//
//    return 0;
//}

//int main() {
//    std::string dir = "/data1/3D_scan/hdRawScan/";
//    std::string path = dir + "depth/";
//    std::string image_path = dir + "bgra/";
//    std::string out_dir = "/home/meidai/下载/kinectfusion1/";
//    std::string trans_dir = out_dir + "trans/";
//    std::string trans_op_dir = out_dir + "trans_op/";
//    std::string first_depth_path = dir + "depth/0.bin";
//    std::string marks_path = dir + "marks/0.bin";
////    std::string marks_path = dir + "marks.bin";
//
//    ////图片分辨率是depth两倍
//    CameraParameter param;
//    param.width = 360;
//    param.height = 640;
//    float fov = 61.459709f;
//    float focal = 0.5f * param.height / tanf(0.5f * fov * M_PI / 180.0f);
//    param.focal_x = focal;
//    param.focal_y = focal;
//    param.principal_x = (param.width - 1) / 2.0f;
//    param.principal_y = (param.height - 1) / 2.0f;
//
//    std::vector<Vec3f> marks;
////    read3DMarks(marks_path, marks);
//    compute3DMarks(first_depth_path, marks_path, marks, param);
//
//    Vec3f max_marks(-1e9, -1e9, -1e9), min_marks(1e9, 1e9, 1e9);
//    for (int i = 0; i < marks.size(); ++i) {
//        for (int j = 0; j < 3; ++j) {
//            if (marks[i](j) < min_marks(j))
//                min_marks(j) = marks[i](j);
//            if (marks[i](j) > max_marks(j))
//                max_marks(j) = marks[i](j);
//        }
//    }
//    float faceRatio = 2.2;
//    float fw = (max_marks(1) - min_marks(1)) * faceRatio;
//    float half_fw = fw * 0.5f;
//    float xCenter = 0.5f * (max_marks(0) + min_marks(0));
//    float yCenter = 0.8f * max_marks(1) + 0.2f * min_marks(1);
//    min_marks(0) = xCenter - half_fw;
//    min_marks(1) = yCenter - half_fw;
//    min_marks(2) -= 15;
//
//    KinectConfig config;
//    config.voxel_scale = fw / (config.volume_size.x - 1);
//    config.truncation_distance = config.voxel_scale * 10;
//    config.volume_origin = min_marks;
//
//    KinectFusion kinectFusion(param, config);
//
//    std::vector<std::string> file_list;
//    readFiles(file_list, path);
//    std::sort(file_list.begin(), file_list.end(), compare);
//
//    time_t t00 = time(0);
//    float *depth = new float[param.height * param.width];
////    float *image = new float[param.height * param.width];
//    for (int i = 0; i < file_list.size(); ++i) {
//        std::string name = path + file_list[i];
//        std::string image_name = image_path + file_list[i];
//        std::cout << name << std::endl;
//        readDepthFloat16(name, depth, param);
////        readImage(image_name, image, param);
////        kinectFusion.process(depth, image);
//        kinectFusion.process(depth);
//    }
////    std::string tmp_path = "/home/meidai/下载/opengl/extract_points.xyz";
////    kinectFusion.extract_points(tmp_path);
////    return 0;
//
//    time_t t01 = time(0);
//    printf("process time: %ld\n", t01 - t00);
//
//    std::string save_path = out_dir + "trajectory_org.obj";
//    save_camera_trajectory(save_path, kinectFusion.pose_list);
////    std::string mesh_obj_path = out_dir + "mesh128_org.obj";
////    kinectFusion.extract_mesh(mesh_obj_path);
////    return 0;
//    std::string mesh_path = out_dir + "mesh128_org.bin";
//    kinectFusion.extract_mesh_bin(mesh_path);
//    for (int i = 0; i < file_list.size(); i++) {
//        std::string trans_path = trans_dir + file_list[i];
//        saveMat(trans_path, kinectFusion.pose_list[i]);
//    }
//
//    time_t t1 = time(0);
//    kinectFusion.optimize();
//    time_t t2 = time(0);
//    printf("optimize time: %ld\n", t2 - t1);
//
//    kinectFusion.reFusion();
//    time_t t3 = time(0);
//    printf("reFusion time: %ld\n", t3 - t2);
//
//    std::string save_path_op = out_dir + "trajectory_op.obj";
//    save_camera_trajectory(save_path_op, kinectFusion.pose_list);
//    std::string mesh_path_op = out_dir + "mesh128_op.bin";
//    kinectFusion.extract_mesh_bin(mesh_path_op);
//    for (int i = 0; i < file_list.size(); i++) {
//        std::string trans_path = trans_op_dir + file_list[i];
//        saveMat(trans_path, kinectFusion.pose_list[i]);
//    }
//
//    return 0;
//}

//int main() {
//    std::string dir = "/data1/3D_scan/yl/frame_data_yl0/";
//    std::string path = dir + "depth/";
//    std::string out_dir = "/home/meidai/下载/kinectfusion2/";
//    std::string trans_dir = dir + "trans/";
//    std::string trans_op_dir = dir + "trans_op/";
//    std::string first_depth_path = dir + "depth/0.bin";
//    std::string marks_path = dir + "marks/0.bin";
////    std::string marks_path = dir + "marks.bin";
//
//    CameraParameter param;
//    param.width = 480;
//    param.height = 640;
//    param.focal_x = 594.818678f;//0.5 * param.height / tan(0.5 * param.fov * M_PI / 180.0)
//    param.focal_y = 594.818678f;
//    param.principal_x = (480 - 1) / 2.0f;
//    param.principal_y = (640 - 1) / 2.0f;
//
//    std::vector<Vec3f> marks;
////    read3DMarks(marks_path, marks);
//    compute3DMarks(first_depth_path, marks_path, marks, param);
//
//    Vec3f max_marks(-1e9, -1e9, -1e9), min_marks(1e9, 1e9, 1e9);
//    for (int i = 0; i < marks.size(); ++i) {
//        for (int j = 0; j < 3; ++j) {
//            if (marks[i](j) < min_marks(j))
//                min_marks(j) = marks[i](j);
//            if (marks[i](j) > max_marks(j))
//                max_marks(j) = marks[i](j);
//        }
//    }
//    float faceRatio = 2.2;
//    float fw = (max_marks(1) - min_marks(1)) * faceRatio;
//    float half_fw = fw * 0.5f;
//    float xCenter = 0.5f * (max_marks(0) + min_marks(0));
//    float yCenter = 0.8f * max_marks(1) + 0.2f * min_marks(1);
//    min_marks(0) = xCenter - half_fw;
//    min_marks(1) = yCenter - half_fw;
//    min_marks(2) -= 15;
//
//    KinectConfig config;
//    config.voxel_scale = fw / (config.volume_size.x - 1);
//    config.truncation_distance = config.voxel_scale * 10;
//    config.volume_origin = min_marks;
//
//    KinectFusion kinectFusion(param, config);
//
//    std::vector<std::string> file_list;
//    readFiles(file_list, path);
//    std::sort(file_list.begin(), file_list.end(), compare);
//
//    time_t t00 = time(0);
//    float *depth = new float[param.height * param.width];
//    for (int i = 0; i < file_list.size(); ++i) {
//        std::string name = path + file_list[i];
//        std::cout << name << std::endl;
//        readDepthFloat16(name, depth, param);
//        kinectFusion.process(depth);
//    }
//
//    time_t t01 = time(0);
//    printf("process time: %ld\n", t01 - t00);
//
//    std::string save_path = out_dir + "trajectory_org.obj";
//    save_camera_trajectory(save_path, kinectFusion.pose_list);
//    std::string mesh_path = out_dir + "mesh128_org.bin";
//    kinectFusion.extract_mesh_bin(mesh_path);
//    for (int i = 0; i < file_list.size(); i++) {
//        std::string trans_path = trans_dir + file_list[i];
//        saveMat(trans_path, kinectFusion.pose_list[i]);
//    }
//
//    time_t t1 = time(0);
//    kinectFusion.optimize();
//    time_t t2 = time(0);
//    printf("optimize time: %ld\n", t2 - t1);
//
//    kinectFusion.reFusion();
//    time_t t3 = time(0);
//    printf("reFusion time: %ld\n", t3 - t2);
//
//    std::string save_path_op = out_dir + "trajectory_op.obj";
//    save_camera_trajectory(save_path_op, kinectFusion.pose_list);
//    std::string mesh_path_op = out_dir + "mesh128_op.bin";
//    kinectFusion.extract_mesh_bin(mesh_path_op);
//    for (int i = 0; i < file_list.size(); i++) {
//        std::string trans_path = trans_op_dir + file_list[i];
//        saveMat(trans_path, kinectFusion.pose_list[i]);
//    }
//
//    return 0;
//}

int main() {
    std::string dir = "/data1/3D_scan/source/";
    std::string path = dir + "depth/";
    std::string image_path = dir + "bgra/";
    std::string init_trans_path = dir + "transform/";
    std::string out_dir = dir;
//    std::string trans_dir = out_dir + "trans/";
    std::string trans_op_dir = out_dir + "trans_op/";
    std::string first_depth_path = dir + "depth/0.bin";
    std::string marks_path = dir + "marks/0.bin";
//    std::string marks_path = dir + "marks.bin";

    CameraParameter param;
    param.width = 360;
    param.height = 640;
    float fov = 61.459709f;
    float focal = 0.5f * param.height / tanf(0.5f * fov * M_PI / 180.0f);
    param.focal_x = focal;
    param.focal_y = focal;

    param.principal_x = (param.width - 1) / 2.0f;
    param.principal_y = (param.height - 1) / 2.0f;

    std::vector<Vec3f> marks;
    compute3DMarks(first_depth_path, marks_path, marks, param, true, 3);

    Vec3f max_marks(-1e9, -1e9, -1e9), min_marks(1e9, 1e9, 1e9);
    for (int i = 0; i < marks.size(); ++i) {
        for (int j = 0; j < 3; ++j) {
            if (marks[i](j) < min_marks(j))
                min_marks(j) = marks[i](j);
            if (marks[i](j) > max_marks(j))
                max_marks(j) = marks[i](j);
        }
    }
    float faceRatio = 2.2;
    float fw = (max_marks(1) - min_marks(1)) * faceRatio;
    float half_fw = fw * 0.5f;
    float xCenter = 0.5f * (max_marks(0) + min_marks(0));
    float yCenter = 0.8f * max_marks(1) + 0.2f * min_marks(1);
    min_marks(0) = xCenter - half_fw;
    min_marks(1) = yCenter - half_fw;
    min_marks(2) -= 15;

    KinectConfig config;
    config.voxel_scale = fw / (config.volume_size.x - 1);
    config.truncation_distance = config.voxel_scale * 10;
    config.volume_origin = min_marks;

    KinectFusion kinectFusion(param, config);

    std::vector<std::string> file_list;
    readFiles(file_list, path);
    std::sort(file_list.begin(), file_list.end(), compare);

    time_t t00 = time(0);
    float *depth = new float[param.height * param.width];
    std::vector<Mat4f> trans_list;
    for (int i = 0; i < file_list.size(); ++i) {
        std::string name = path + file_list[i];
        std::string trans_name = init_trans_path + file_list[i];
        std::cout << name << std::endl;
        Mat4f trans;
        readDepthFloat16(name, depth, param);
        readTrans(trans_name, trans);
        kinectFusion.process(depth, trans);
        trans_list.push_back(trans);
    }
    time_t t01 = time(0);
    printf("process time: %ld\n", t01 - t00);

//    std::string save_path = out_dir + "trajectory_ios.obj";
//    save_camera_trajectory(save_path, kinectFusion.pose_list);
//    std::string mesh_obj_path = out_dir + "mesh128_org.obj";
//    kinectFusion.extract_mesh(mesh_obj_path);
//    return 0;
    std::string mesh_path = out_dir + "mesh128_org.bin";
    kinectFusion.extract_mesh_bin(mesh_path);
//    for (int i = 0; i < file_list.size(); i++) {
//        std::string trans_path = trans_dir + file_list[i];
//        saveMat(trans_path, kinectFusion.pose_list[i]);
//    }

    time_t t1 = time(0);
    kinectFusion.optimize();
    time_t t2 = time(0);
    printf("optimize time: %ld\n", t2 - t1);

    kinectFusion.reFusion();
    time_t t3 = time(0);
    printf("reFusion time: %ld\n", t3 - t2);

    std::string save_path_op = out_dir + "trajectory_op.obj";
    save_camera_trajectory(save_path_op, kinectFusion.pose_list);
    std::string mesh_path_op = out_dir + "mesh128_op.obj";
//    kinectFusion.extract_mesh_bin(mesh_path_op);
    kinectFusion.extract_mesh(mesh_path_op);

    Mat4f I;
    I.setIdentity();
    I(2, 2) = -1;
    for (int i = 0; i < file_list.size(); i++) {
        std::string trans_path = trans_op_dir + file_list[i];
        Mat4f trans = I * kinectFusion.pose_list[i] * I;
        saveMat(trans_path, trans);
    }

    return 0;
}