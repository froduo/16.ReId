#include "reidextractor.h"
#include "backgroundmodel.h"
#include "reiddeep.h"

#include <algorithm>
#include <cmath>
#include <vector>

ReIDExtractor::ReIDExtractor()
{
}

// ============================================================================
// 分量权重
// ============================================================================
void ReIDExtractor::setComponentWeights(float color, float texture, float shape)
{
    color   = std::max(0.0f, color);
    texture = std::max(0.0f, texture);
    shape   = std::max(0.0f, shape);

    float sum = color + texture + shape;
    if (sum <= 0.0f) {
        // 与 ReIDExtractor 默认权重保持一致：颜色权重下调以提升跨朝向鲁棒性；
        // 身形维度由 2 维扩展为 6 维多模态几何后，其区分力提升，权重同步上调。
        color = 0.30f; texture = 0.35f; shape = 0.35f;
        sum = 1.0f;
    }
    m_wColor   = color / sum;
    m_wTexture = texture / sum;
    m_wShape   = shape / sum;
}

// ============================================================================
// 颜色分量：HSV 三维直方图，L2 归一化，返回 1×COLOR_HALF
// ============================================================================
cv::Mat ReIDExtractor::computeColorHistogram(const cv::Mat &roi, const cv::Mat &mask)
{
    if (roi.empty() || roi.rows < 1 || roi.cols < 1)
        return cv::Mat::zeros(1, COLOR_HALF, CV_32F);

    cv::Mat hsv;
    if (roi.channels() == 3) {
        cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);
    } else if (roi.channels() == 4) {
        cv::cvtColor(roi, hsv, cv::COLOR_BGRA2BGR);
        cv::cvtColor(hsv, hsv, cv::COLOR_BGR2HSV);
    } else {
        cv::cvtColor(roi, hsv, cv::COLOR_GRAY2BGR);
        cv::cvtColor(hsv, hsv, cv::COLOR_BGR2HSV);
    }

    int histSize[] = {H_BINS, S_BINS, V_BINS};
    float hRanges[] = {0, 180};
    float sRanges[] = {0, 256};
    float vRanges[] = {0, 256};
    const float *ranges[] = {hRanges, sRanges, vRanges};
    int channels[] = {0, 1, 2};

    cv::Mat hist;
    // mask 非空时只统计人体前景像素，剔除背景
    cv::Mat useMask = (!mask.empty() && mask.size() == hsv.size()) ? mask : cv::Mat();
    cv::calcHist(&hsv, 1, channels, useMask, hist, 3, histSize, ranges);

    cv::normalize(hist, hist, 1.0, 0.0, cv::NORM_L2);
    hist = hist.reshape(1, 1);
    hist.convertTo(hist, CV_32F);
    return hist;
}

// ============================================================================
// 纹理分量：灰度梯度方向直方图（取模 180°，消除正负向），L2 归一化
// ============================================================================
cv::Mat ReIDExtractor::computeTextureHistogram(const cv::Mat &roi, const cv::Mat &mask)
{
    if (roi.empty() || roi.cols < 3 || roi.rows < 3)
        return cv::Mat::zeros(1, TEX_BINS, CV_32F);

    const bool useMask = (!mask.empty() && mask.size() == roi.size());

    cv::Mat gray;
    if (roi.channels() == 3) {
        cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    } else if (roi.channels() == 4) {
        cv::cvtColor(roi, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = roi;
    }

    cv::Mat gx, gy;
    cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, gy, CV_32F, 0, 1, 3);

    cv::Mat mag, ang;
    cv::cartToPolar(gx, gy, mag, ang, true); // ang ∈ [0,360)

    cv::Mat hist = cv::Mat::zeros(1, TEX_BINS, CV_32F);
    for (int y = 0; y < gray.rows; ++y) {
        const float *m = mag.ptr<float>(y);
        const float *a = ang.ptr<float>(y);
        for (int x = 0; x < gray.cols; ++x) {
            if (useMask && mask.at<uchar>(y, x) == 0) continue; // 跳过背景像素
            const float mm = m[x];
            if (mm < 1e-3f) continue;
            float aa = std::fmod(a[x], 180.0f);
            if (aa < 0.0f) aa += 180.0f;
            int b = static_cast<int>(aa / 180.0f * TEX_BINS);
            if (b < 0) b = 0;
            if (b >= TEX_BINS) b = TEX_BINS - 1;
            hist.at<float>(0, b) += mm;
        }
    }

    cv::normalize(hist, hist, 1.0, 0.0, cv::NORM_L2);
    return hist;
}

// ============================================================================
// 身形分量：6 维多模态几何/体积代理
//   俯视场景下人被压缩为「头顶+肩部」投影，颜色/纹理区分度下降，
//   因此补充与朝向弱相关、与身材强相关的几何与体积代理维度：
//     0 长宽比(w/h)        1 归一化高度(h/H)    2 归一化宽度(w/W)
//     3 面积占比           4 前景填充率(体积代理) 5 肩部宽度比(肩宽/h)
// ============================================================================
cv::Mat ReIDExtractor::computeShapeVec(const cv::Mat &frame, const cv::Rect &box,
                                       const cv::Mat &mask, int fgPixels)
{
    cv::Mat s = cv::Mat::zeros(1, SHAPE_DIM, CV_32F);
    const float w  = static_cast<float>(std::max(1, box.width));
    const float h  = static_cast<float>(std::max(1, box.height));
    const float fw = static_cast<float>(std::max(1, frame.cols));
    const float fh = static_cast<float>(std::max(1, frame.rows));

    // 注意：s 为 1×SHAPE_DIM 的行向量，索引必须写作 at<float>(0, i)（行0、第i列）；
    //       写成 at<float>(i, 0) 会越界写坏堆（heap-buffer-overflow）。
    s.at<float>(0, 0) = w / h;                    // 长宽比
    s.at<float>(0, 1) = h / fh;                   // 归一化高度
    s.at<float>(0, 2) = w / fw;                   // 归一化宽度
    s.at<float>(0, 3) = (w * h) / (fw * fh);      // 面积占比（体积代理之一）

    // 前景填充率 = 前景像素 / 外接框面积（无掩码时退化为 1，不参与区分）
    s.at<float>(0, 4) = (fgPixels > 0)
            ? std::min(1.0f, static_cast<float>(fgPixels) / (w * h)) : 1.0f;

    // 肩部宽度比：前景在「上方 40% 行」内的最大连续宽度 / 框高
    float shoulder = w;
    if (!mask.empty() && mask.rows > 0 && mask.cols > 0) {
        const int topRows = std::max(1, static_cast<int>(mask.rows * 0.4f));
        int maxRun = 1;
        for (int y = 0; y < topRows && y < mask.rows; ++y) {
            const uchar *row = mask.ptr<uchar>(y);
            int run = 0;
            for (int x = 0; x < mask.cols; ++x) {
                if (row[x]) { ++run; maxRun = std::max(maxRun, run); }
                else run = 0;
            }
        }
        shoulder = static_cast<float>(maxRun);
    }
    s.at<float>(0, 5) = shoulder / h;             // 肩宽/身高

    return s;
}

// ============================================================================
// 单帧特征：颜色 + 纹理 + 身形 → 1×TOTAL_DIM
// ============================================================================
cv::Mat ReIDExtractor::extract(const cv::Mat &frame, const cv::Rect &personBox)
{
    if (frame.empty() || personBox.empty())
        return cv::Mat();

    cv::Rect safeBox = personBox & cv::Rect(0, 0, frame.cols, frame.rows);
    if (safeBox.empty() || safeBox.width < 2 || safeBox.height < 2)
        return cv::Mat();

    // ===== 深度 ReID 后端：直接送入 OSNet（可选「头框→人体框」扩展）=====
    if (deepEnabled()) {
        cv::Rect use = safeBox;
        if (m_boxExpand > 1.0f) {
            const int cx = use.x + use.width / 2;
            const int cy = use.y + use.height / 2;
            const int w = static_cast<int>(use.width * m_boxExpand);
            const int h = static_cast<int>(use.height * m_boxExpand);
            use = cv::Rect(cx - w / 2, cy - h / 2, w, h) & cv::Rect(0, 0, frame.cols, frame.rows);
        }
        cv::Mat deepFeat = m_deep->embed(frame, use);
        if (!deepFeat.empty()) return deepFeat;
        // 深度推理失败 → 回退手工特征（保证流程不中断）
    }

    // ===== 背景扣除：获取人体前景掩码，用于剔除背景像素 =====
    cv::Mat roiMask;
    if (m_bgModel && m_bgModel->ready()) {
        cv::Mat fullMask = m_bgModel->foregroundMask(frame);
        if (!fullMask.empty()) {
            roiMask = fullMask(safeBox).clone();
        }
    }

    cv::Mat personCrop = frame(safeBox);
    cv::Rect usedBox = safeBox;

    // 用前景掩码收紧人体框（去掉框内多余背景）
    if (!roiMask.empty()) {
        std::vector<cv::Point> pts;
        cv::findNonZero(roiMask, pts);
        if (!pts.empty()) {
            const cv::Rect r = cv::boundingRect(pts);
            const double areaRatio = static_cast<double>(r.area()) /
                                     static_cast<double>(safeBox.area());
            if (r.width >= 4 && r.height >= 4 && areaRatio >= 0.20) {
                personCrop = personCrop(r);
                roiMask = roiMask(r);
                usedBox = cv::Rect(safeBox.x + r.x, safeBox.y + r.y, r.width, r.height);
            } else {
                roiMask = cv::Mat(); // 前景过小/不可靠 → 回退为整框
            }
        } else {
            roiMask = cv::Mat();
        }
    }

    const int W = personCrop.cols;
    const int H = personCrop.rows;
    if (W < 2 || H < 2) return cv::Mat();

    // 上/下半身划分
    int halfH = H / 2;
    if (halfH < 1) halfH = 1;
    if (halfH >= H) halfH = H - 1;

    cv::Rect upperRegion(0, 0, W, halfH);
    cv::Rect lowerRegion(0, halfH, W, H - halfH);

    cv::Mat upperCrop = personCrop(upperRegion);
    cv::Mat lowerCrop = personCrop(lowerRegion);

    cv::Mat upperMask, lowerMask;
    if (!roiMask.empty()) {
        upperMask = roiMask(upperRegion);
        lowerMask = roiMask(lowerRegion);
    }

    // 颜色（上/下半身）
    cv::Mat cUp = computeColorHistogram(upperCrop, upperMask);
    cv::Mat cLo = computeColorHistogram(lowerCrop, lowerMask);
    // 纹理（上/下半身）
    cv::Mat tUp = computeTextureHistogram(upperCrop, upperMask);
    cv::Mat tLo = computeTextureHistogram(lowerCrop, lowerMask);
    // 身形（6 维多模态几何/体积代理；用收紧后的框 + 前景掩码）
    const int fgPixels = roiMask.empty() ? 0 : cv::countNonZero(roiMask);
    cv::Mat shape = computeShapeVec(frame, usedBox, roiMask, fgPixels);

    cv::Mat feat;
    cv::hconcat(std::vector<cv::Mat>{cUp, cLo, tUp, tLo, shape}, feat);
    feat.convertTo(feat, CV_32F);
    return feat;
}

// ============================================================================
// 多帧累积：逐元素平均 + 分量归一化（尺寸/通道不一致的自动跳过）
// ============================================================================
cv::Mat ReIDExtractor::averageFeatures(const QVector<cv::Mat> &features)
{
    QVector<cv::Mat> valid;
    for (const cv::Mat &f : features) {
        if (!f.empty()) valid.append(f);
    }
    if (valid.isEmpty()) return cv::Mat();

    const cv::Mat base = valid.first().reshape(1, 1);
    const cv::Size sz = base.size();
    const int ch = base.channels();

    cv::Mat sum;
    base.convertTo(sum, CV_32F);
    int used = 1;
    for (int i = 1; i < valid.size(); ++i) {
        cv::Mat f = valid[i].reshape(1, 1);
        if (f.size() != sz || f.channels() != ch) continue; // 跳过不一致
        cv::Mat tmp;
        f.convertTo(tmp, CV_32F);
        sum += tmp;
        used++;
    }
    if (used > 1) {
        sum /= static_cast<float>(used);
    }

    // 归一化：完整手工 bundle 按分量归一化；其它维度（如深度 512 维）整体 L2 归一化
    if (sum.cols == TOTAL_DIM) {
        cv::Mat colorBlock = sum.colRange(0, COLOR_DIM);
        cv::normalize(colorBlock, colorBlock, 1.0, 0.0, cv::NORM_L2);
        cv::Mat texBlock = sum.colRange(COLOR_DIM, COLOR_DIM + TEX_DIM);
        cv::normalize(texBlock, texBlock, 1.0, 0.0, cv::NORM_L2);
    } else {
        sum = l2normalize(sum);
    }

    return sum;
}

// ============================================================================
// 各分量相似度
// ============================================================================
float ReIDExtractor::compareColor(const cv::Mat &a, const cv::Mat &b) const
{
    if (a.empty() || b.empty() || a.cols != b.cols || a.cols < 2)
        return 0.0f;

    const int half = a.cols / 2;
    const cv::Mat a1 = a.colRange(0, half),     b1 = b.colRange(0, half);
    const cv::Mat a2 = a.colRange(half, a.cols), b2 = b.colRange(half, b.cols);

    const double d1 = cv::compareHist(a1, b1, cv::HISTCMP_BHATTACHARYYA);
    const double d2 = cv::compareHist(a2, b2, cv::HISTCMP_BHATTACHARYYA);
    double d = 0.55 * d1 + 0.45 * d2;   // 上身颜色更具区分度
    if (!std::isfinite(d)) d = 1.0;
    return static_cast<float>(std::max(0.0, std::min(1.0, 1.0 - d)));
}

float ReIDExtractor::compareTexture(const cv::Mat &a, const cv::Mat &b) const
{
    if (a.empty() || b.empty() || a.cols != b.cols || a.cols < 2)
        return 0.5f; // 无法比较时给中性值，避免误判

    const int half = a.cols / 2;
    const cv::Mat a1 = a.colRange(0, half),     b1 = b.colRange(0, half);
    const cv::Mat a2 = a.colRange(half, a.cols), b2 = b.colRange(half, b.cols);

    const double d1 = cv::compareHist(a1, b1, cv::HISTCMP_BHATTACHARYYA);
    const double d2 = cv::compareHist(a2, b2, cv::HISTCMP_BHATTACHARYYA);
    double d = 0.5 * d1 + 0.5 * d2;
    if (!std::isfinite(d)) d = 1.0;
    return static_cast<float>(std::max(0.0, std::min(1.0, 1.0 - d)));
}

float ReIDExtractor::compareShape(const cv::Mat &a, const cv::Mat &b) const
{
    if (a.empty() || b.empty()) return 0.5f;

    const cv::Mat ai = a.reshape(1, 1);
    const cv::Mat bi = b.reshape(1, 1);
    const int n = static_cast<int>(std::min(ai.total(), bi.total()));
    if (n < 2) return 0.5f;

    // 各几何维度权重：长宽比/肩宽更能刻画身材，体积类维度权重略低
    static const float wts[SHAPE_DIM] = {0.28f, 0.18f, 0.16f, 0.12f, 0.08f, 0.18f};

    auto relSim = [](float x, float y) -> float {
        const float m = std::max(std::fabs(x), std::fabs(y));
        if (m < 1e-6f) return 1.0f;
        return std::max(0.0f, 1.0f - std::fabs(x - y) / m);
    };

    float sum = 0.0f, wsum = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float w = wts[i];
        sum  += w * relSim(ai.at<float>(0, i), bi.at<float>(0, i));
        wsum += w;
    }
    if (wsum <= 1e-6f) return 0.5f;
    return std::max(0.0f, std::min(1.0f, sum / wsum));
}

// 上下半身交换（等效 180° 旋转）——前置声明，供分量评分复用
static cv::Mat rot180Feature(const cv::Mat &f, int colorDim, int texDim);

// ============================================================================
// 分量级评分（供动态加权 / 全局匹配使用）
// 已内置 180° 旋转不变：对 (a,b) 与其旋转版本逐分量取较大者。
// ============================================================================
ComponentScores ReIDExtractor::compareComponents(const cv::Mat &feat1, const cv::Mat &feat2) const
{
    ComponentScores sc;
    if (feat1.empty() || feat2.empty()) return sc;

    const cv::Mat a = feat1.reshape(1, 1);
    const cv::Mat b = feat2.reshape(1, 1);
    if (a.total() != b.total() || a.channels() != b.channels()) return sc;

    if (a.cols == TOTAL_DIM) {
        sc.color   = compareColor(a.colRange(0, COLOR_DIM), b.colRange(0, COLOR_DIM));
        sc.texture = compareTexture(a.colRange(COLOR_DIM, COLOR_DIM + TEX_DIM),
                                    b.colRange(COLOR_DIM, COLOR_DIM + TEX_DIM));
        sc.shape   = compareShape(a.colRange(COLOR_DIM + TEX_DIM, TOTAL_DIM),
                                  b.colRange(COLOR_DIM + TEX_DIM, TOTAL_DIM));

        // 旋转不变：上车/下车时同人旋转约 180°，逐分量取更优
        const cv::Mat bRot = rot180Feature(b, COLOR_DIM, TEX_DIM);
        ComponentScores sr;
        sr.color   = compareColor(a.colRange(0, COLOR_DIM), bRot.colRange(0, COLOR_DIM));
        sr.texture = compareTexture(a.colRange(COLOR_DIM, COLOR_DIM + TEX_DIM),
                                    bRot.colRange(COLOR_DIM, COLOR_DIM + TEX_DIM));
        sr.shape   = compareShape(a.colRange(COLOR_DIM + TEX_DIM, TOTAL_DIM),
                                  bRot.colRange(COLOR_DIM + TEX_DIM, TOTAL_DIM));
        sc.color   = std::max(sc.color,   sr.color);
        sc.texture = std::max(sc.texture, sr.texture);
        sc.shape   = std::max(sc.shape,   sr.shape);
    } else {
        sc.color   = compareColor(a, b);
        sc.texture = 0.5f;
        sc.shape   = 0.5f;
    }
    return sc;
}

/// 上下半身交换（等效 180° 旋转）：颜色与纹理块各自交换上/下半身
static cv::Mat rot180Feature(const cv::Mat &f, int colorDim, int texDim)
{
    cv::Mat out = f.clone();
    const int cHalf = colorDim / 2;
    const int tHalf = texDim / 2;
    f.colRange(cHalf, colorDim).copyTo(out.colRange(0, cHalf));
    f.colRange(0, cHalf).copyTo(out.colRange(cHalf, colorDim));
    f.colRange(colorDim + tHalf, colorDim + texDim).copyTo(out.colRange(colorDim, colorDim + tHalf));
    f.colRange(colorDim, colorDim + tHalf).copyTo(out.colRange(colorDim + tHalf, colorDim + texDim));
    return out;
}

float ReIDExtractor::compareWithWeights(const cv::Mat &feat1, const cv::Mat &feat2,
                                        float wColor, float wTexture, float wShape) const
{
    if (feat1.empty() || feat2.empty()) return 0.0f;

    const cv::Mat a = feat1.reshape(1, 1);
    const cv::Mat b = feat2.reshape(1, 1);
    if (a.total() != b.total() || a.channels() != b.channels()) return 0.0f;

    float sim = compareComponents(a, b).fused(wColor, wTexture, wShape);

    // 旋转不变：上车/下车时同人旋转约 180°，交换上下半身后再取较大者
    if (a.cols == TOTAL_DIM) {
        const cv::Mat bRot = rot180Feature(b, COLOR_DIM, TEX_DIM);
        sim = std::max(sim, compareComponents(a, bRot).fused(wColor, wTexture, wShape));
    }
    return std::max(0.0f, std::min(1.0f, sim));
}

float ReIDExtractor::compare(const cv::Mat &feat1, const cv::Mat &feat2)
{
    // 深度后端：余弦相似度（L2 归一化特征点积）
    if (deepEnabled())
        return m_deep->compare(feat1, feat2);
    return compareWithWeights(feat1, feat2, m_wColor, m_wTexture, m_wShape);
}

// ============================================================================
// 深度 ReID 后端接入与工具
// ============================================================================
void ReIDExtractor::setDeepReid(DeepReID *deep, bool use)
{
    m_deep = deep;
    m_useDeep = use;
}

bool ReIDExtractor::deepEnabled() const
{
    return m_useDeep && m_deep != nullptr && m_deep->ready();
}

int ReIDExtractor::dim() const
{
    return deepEnabled() ? DeepReID::featureDim() : TOTAL_DIM;
}

cv::Mat ReIDExtractor::l2normalize(const cv::Mat &v)
{
    if (v.empty()) return v;
    cv::Mat x = v.reshape(1, 1).clone();
    x.convertTo(x, CV_32F);
    const double n = cv::norm(x, cv::NORM_L2);
    if (n > 1e-12) x /= static_cast<float>(n);
    return x;
}

// ============================================================================
// 多帧时序融合（EMA）：越新的帧权重越高
// ============================================================================
cv::Mat ReIDExtractor::temporalFuse(const QVector<cv::Mat> &features, float alpha)
{
    if (alpha <= 0.0f || alpha >= 1.0f)
        return averageFeatures(features);

    cv::Mat fused;
    int used = 0;
    for (const cv::Mat &f : features) {
        if (f.empty()) continue;
        cv::Mat ff = f.reshape(1, 1);
        ff.convertTo(ff, CV_32F);
        if (fused.empty()) { fused = ff.clone(); used = 1; continue; }
        if (fused.size() != ff.size() || fused.channels() != ff.channels()) continue;
        cv::Mat out = ff * alpha + fused * (1.0f - alpha);
        fused = out;
        ++used;
    }
    if (fused.empty()) return cv::Mat();

    if (fused.cols == TOTAL_DIM) {
        cv::Mat colorBlock = fused.colRange(0, COLOR_DIM);
        cv::normalize(colorBlock, colorBlock, 1.0, 0.0, cv::NORM_L2);
        cv::Mat texBlock = fused.colRange(COLOR_DIM, COLOR_DIM + TEX_DIM);
        cv::normalize(texBlock, texBlock, 1.0, 0.0, cv::NORM_L2);
    } else {
        // 深度特征：余弦相似度要求 L2 归一化
        fused = l2normalize(fused);
    }
    return fused;
}

// ============================================================================
// 时序融合缓存（按 key 维护 EMA 融合特征）
// ============================================================================
cv::Mat TemporalFeatureCache::update(int key, const cv::Mat &feat)
{
    if (feat.empty()) return get(key);

    cv::Mat ff = feat.reshape(1, 1);
    ff.convertTo(ff, CV_32F);

    Entry &e = m_entries[key];
    if (e.fused.empty() || e.fused.size() != ff.size() || e.fused.channels() != ff.channels()) {
        e.fused = ff.clone();
        e.n = 1;
    } else {
        cv::Mat out = ff * m_alpha + e.fused * (1.0f - m_alpha);
        e.fused = out;
        ++e.n;
    }
    return e.fused.clone();
}

cv::Mat TemporalFeatureCache::get(int key) const
{
    auto it = m_entries.find(key);
    return (it != m_entries.end()) ? it.value().fused.clone() : cv::Mat();
}

int TemporalFeatureCache::count(int key) const
{
    auto it = m_entries.find(key);
    return (it != m_entries.end()) ? it.value().n : 0;
}
