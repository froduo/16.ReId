// ReID 描述子自检：验证编译后的特征维度与新描述子行为
// 编译：见 workspace 根目录下的命令（直接编译源码，确保使用最新代码）
#include "reidextractor.h"

#include <cstdio>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv)
{
    printf("ReIDExtractor::featureDim() = %d\n", ReIDExtractor::featureDim());

    ReIDExtractor ex;
    printf("weights: color=%.2f texture=%.2f shape=%.2f\n",
           ex.weightColor(), ex.weightTexture(), ex.weightShape());

    if (argc < 3) {
        printf("用法: reid_selftest <imgA> <imgB>\n");
        return 0;
    }

    cv::Mat a = cv::imread(argv[1]);
    cv::Mat b = cv::imread(argv[2]);
    if (a.empty() || b.empty()) {
        printf("读取图像失败\n");
        return 1;
    }

    // 以整图作为人体框（这些是人物居中的裁剪图）
    cv::Mat fa = ex.extract(a, cv::Rect(0, 0, a.cols, a.rows));
    cv::Mat fb = ex.extract(b, cv::Rect(0, 0, b.cols, b.rows));

    printf("featA = %d x %d   featB = %d x %d\n", fa.rows, fa.cols, fb.rows, fb.cols);
    printf("sim(A,B) = %.4f\n", ex.compare(fa, fb));
    return 0;
}
