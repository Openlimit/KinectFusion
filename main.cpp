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

int main() {
    std::string path = "/data1/3D_scan/yl/frame_data_yl1/depth/";
    std::string out_dir = "/home/meidai/下载/kinectfusion/";
    std::string marks_path = "/data1/3D_scan/yl/frame_data_yl1/marks.bin";
    std::string out_marks_path = out_dir + "marks.xyz";

    CameraParameter param;
    param.width = 480;
    param.height = 640;
    param.focal_x = 594.818678f;
    param.focal_y = 594.818678f;
    param.principal_x = (480 - 1) / 2.0f;
    param.principal_y = (640 - 1) / 2.0f;

    std::vector<Vec3f> marks;
    read3DMarks(marks_path, marks);
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

    float *depth = new float[param.height * param.width];
    for (int i = 0; i < file_list.size(); ++i) {
        std::string name = path + file_list[i];
        std::cout << name << std::endl;
        readDepthFloat16(name, depth, param);
        kinectFusion.process(depth);
    }

    std::string save_path = out_dir + "trajectory_org.obj";
    save_camera_trajectory(save_path, kinectFusion.pose_list);
    std::string mesh_path = out_dir + "mesh128_org.obj";
    kinectFusion.extract_mesh(mesh_path);

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
    kinectFusion.extract_mesh(mesh_path_op);

    return 0;
}