// 临时诊断工具：用 TensorRT engine 对图片做检测，打印原始候选（低阈值），
// 用于对比 best.pt / ONNX 的推理结果，判断 engine 是否与模型行为一致。
#include "trt_yolo.h"
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <vector>

int main(int argc, char **argv)
{
    if (argc < 3) {
        std::printf("usage: %s <engine> <img...>\n", argv[0]);
        return 1;
    }

    TrtLogger logger;
    TrtYolo yolo(argv[1], logger, 0.001f);
    yolo.setLetterbox(true);

    for (int i = 2; i < argc; ++i) {
        cv::Mat img = cv::imread(argv[i]);
        if (img.empty()) {
            std::printf("read fail: %s\n", argv[i]);
            continue;
        }
        yolo.preprocess(img);
        yolo.infer();
        std::vector<Detection> dets;
        yolo.postprocessAll(dets, 0.001f);
        std::printf("%s : dets=%zu\n", argv[i], dets.size());
        for (const auto &d : dets) {
            std::printf("   cls=%d score=%.4f box=(%d,%d,%d,%d)\n",
                        d.class_id, d.score, d.x, d.y, d.w, d.h);
        }
    }
    return 0;
}
