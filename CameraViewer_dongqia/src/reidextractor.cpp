#include "reidextractor.h"

ReIDExtractor::ReIDExtractor()
{
}

cv::Mat ReIDExtractor::computeColorHistogram(const cv::Mat &roi)
{
    // Convert to HSV color space
    cv::Mat hsv;
    cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);

    // Define histogram parameters
    int histSize[] = {H_BINS, S_BINS, V_BINS};
    float hRanges[] = {0, 180};
    float sRanges[] = {0, 256};
    float vRanges[] = {0, 256};
    const float *ranges[] = {hRanges, sRanges, vRanges};
    int channels[] = {0, 1, 2};

    // Compute 3D HSV histogram
    cv::Mat hist;
    cv::calcHist(&hsv, 1, channels, cv::Mat(), hist, 3, histSize, ranges);

    // Normalize histogram (L2 norm)
    cv::normalize(hist, hist, 1.0, 0.0, cv::NORM_L2);

    // Reshape to 1D column vector
    hist = hist.reshape(1, 1);

    return hist;
}

cv::Mat ReIDExtractor::extract(const cv::Mat &frame, const cv::Rect &personBox)
{
    if (frame.empty() || personBox.empty())
        return cv::Mat();

    // Clamp personBox to frame boundaries
    cv::Rect safeBox = personBox & cv::Rect(0, 0, frame.cols, frame.rows);
    if (safeBox.empty())
        return cv::Mat();

    cv::Mat personCrop = frame(safeBox);

    // Split into upper body (top 50%) and lower body (bottom 50%)
    int halfHeight = safeBox.height / 2;

    cv::Rect upperRegion(0, 0, safeBox.width, halfHeight);
    cv::Rect lowerRegion(0, halfHeight, safeBox.width, safeBox.height - halfHeight);

    cv::Mat upperCrop = personCrop(upperRegion);
    cv::Mat lowerCrop = personCrop(lowerRegion);

    // Compute color histograms for each region
    cv::Mat upperHist = computeColorHistogram(upperCrop);
    cv::Mat lowerHist = computeColorHistogram(lowerCrop);

    // Concatenate upper and lower body features into a single feature vector
    cv::Mat feature;
    cv::hconcat(upperHist, lowerHist, feature);

    return feature;
}

float ReIDExtractor::compare(const cv::Mat &feat1, const cv::Mat &feat2)
{
    if (feat1.empty() || feat2.empty())
        return 0.0f;

    // Split concatenated features into upper and lower body parts
    int featureLen = feat1.cols;
    int halfLen = featureLen / 2;

    // Upper body comparison
    cv::Mat upper1 = feat1.colRange(0, halfLen);
    cv::Mat upper2 = feat2.colRange(0, halfLen);

    // Lower body comparison
    cv::Mat lower1 = feat1.colRange(halfLen, featureLen);
    cv::Mat lower2 = feat2.colRange(halfLen, featureLen);

    // Compute Bhattacharyya distance for each part
    // cv::compareHist returns distance: 0 = identical, 1 = completely different
    double upperDist = cv::compareHist(upper1, upper2, cv::HISTCMP_BHATTACHARYYA);
    double lowerDist = cv::compareHist(lower1, lower2, cv::HISTCMP_BHATTACHARYYA);

    // Weight upper body slightly more (clothes color is more distinctive)
    double weightedDist = 0.55 * upperDist + 0.45 * lowerDist;

    // Convert distance [0, 1] to similarity [1, 0] -> [0, 1]
    float similarity = static_cast<float>(1.0 - weightedDist);

    // Clamp to [0, 1]
    return std::max(0.0f, std::min(1.0f, similarity));
}
