#ifndef REIDEXTRACTOR_H
#define REIDEXTRACTOR_H

#include <opencv2/opencv.hpp>

class ReIDExtractor
{
public:
    ReIDExtractor();

    cv::Mat extract(const cv::Mat &frame, const cv::Rect &personBox);
    float compare(const cv::Mat &feat1, const cv::Mat &feat2);

private:
    static constexpr int H_BINS = 30;
    static constexpr int S_BINS = 32;
    static constexpr int V_BINS = 32;

    cv::Mat computeColorHistogram(const cv::Mat &roi);
};

#endif // REIDEXTRACTOR_H
