// 深度 ReID(OSNet) C++ 自检：组质心余弦相似度，验证 V41/V43(同人) 与 V45(异人) 的可分性
// 用法: deepreid_selftest <onnx> <dir1> <dir2> <dir3>
#include "reiddeep.h"

#include <cstdio>
#include <dirent.h>
#include <string>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>

static std::vector<std::string> listJpg(const std::string &dir)
{
    std::vector<std::string> out;
    DIR *d = opendir(dir.c_str());
    if (!d) return out;
    struct dirent *e;
    while ((e = readdir(d))) {
        std::string n = e->d_name;
        if (n.size() > 4 && (n.substr(n.size() - 4) == ".jpg" || n.substr(n.size() - 4) == ".JPG"))
            out.push_back(dir + "/" + n);
    }
    closedir(d);
    std::sort(out.begin(), out.end());
    return out;
}

int main(int argc, char **argv)
{
    if (argc < 5) {
        printf("用法: deepreid_selftest <onnx> <dirA> <dirB> <dirC>\n");
        return 0;
    }
    DeepReID deep;
    if (!deep.load(argv[1])) { printf("模型加载失败\n"); return 1; }
    printf("模型已加载: %s  dim=%d\n", deep.modelPath().toStdString().c_str(), DeepReID::featureDim());

    std::vector<std::string> dirs = {argv[2], argv[3], argv[4]};
    std::vector<cv::Mat> cents;

    for (const auto &dir : dirs) {
        auto files = listJpg(dir);
        cv::Mat sum;
        int n = 0;
        for (const auto &f : files) {
            cv::Mat img = cv::imread(f);
            if (img.empty()) continue;
            // 取中心 62.5% 作为近似人体框（与 saveVisitorFrame 的对称 padding 对应）
            const int h = img.rows, w = img.cols;
            cv::Rect box(int(w * 0.1875), int(h * 0.1875), int(w * 0.625), int(h * 0.625));
            box &= cv::Rect(0, 0, w, h);
            cv::Mat feat = deep.embed(img, box);
            if (feat.empty()) continue;
            if (sum.empty()) sum = cv::Mat::zeros(1, feat.cols, CV_32F);
            sum += feat;
            n++;
        }
        if (n > 0) {
            sum /= static_cast<float>(n);
            const double nn = cv::norm(sum, cv::NORM_L2);
            if (nn > 1e-12) sum /= static_cast<float>(nn);
        }
        printf("组 %s: %d 帧\n", dir.c_str(), n);
        cents.push_back(sum);
    }

    for (size_t i = 0; i < cents.size(); ++i)
        for (size_t j = i + 1; j < cents.size(); ++j)
            printf("  %s <-> %s : %.4f\n", dirs[i].c_str(), dirs[j].c_str(),
                   deep.compare(cents[i], cents[j]));
    return 0;
}
