#ifndef REIDEXTRACTOR_H
#define REIDEXTRACTOR_H

#include <QVector>
#include <QMap>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>

class BackgroundModel;
class DeepReID;

/**
 * @brief 单帧特征的分量相似度（用于「动态加权」）
 */
struct ComponentScores {
    float color{0.0f};     // 颜色（上/下半身 HSV 直方图）
    float texture{0.0f};   // 纹理（梯度方向直方图）
    float shape{0.0f};     // 身形（多模态几何：长宽比/肩宽/体积/填充率…）

    /// 按给定权重融合（权重内部归一化）
    inline float fused(float wc, float wt, float ws) const {
        const float sum = wc + wt + ws;
        const float d = (sum > 1e-6f) ? sum : 1.0f;
        float s = (wc * color + wt * texture + ws * shape) / d;
        if (!(s == s)) s = 0.0f;                      // NaN 保护
        return std::max(0.0f, std::min(1.0f, s));
    }
};

/**
 * @brief 行人重识别特征提取器（多模态融合版）
 *
 * 单帧特征 = 颜色 + 纹理 + 身形 三类分量拼接为一个 1×N 的 CV_32F 行向量：
 *   [ 颜色: 上半身HSV直方图 | 下半身HSV直方图 ]               (2 * H*S*V)
 *   [ 纹理: 上半身梯度方向直方图 | 下半身梯度方向直方图 ]      (2 * TEX_BINS)
 *   [ 身形: 6 维几何/体积代理 ]                                (SHAPE_DIM)
 *
 * 俯视场景下人体被压缩为「头顶+肩部」投影，外观区分度天然偏低，因此在原有
 * 颜色/纹理基础上**扩展身形几何维度**（长宽比、归一化高/宽、面积占比、前景
 * 填充率、肩部宽度），并支持：
 *   - 分量级评分 compareComponents()（供动态加权/全局匹配使用）；
 *   - 多帧时序融合 temporalFuse() / TemporalFeatureCache（EMA 递推平均）。
 *
 * 相似度 = 三类分量相似度的加权融合（权重可调），各分量内部再做上下半身加权：
 *   颜色/纹理：Bhattacharyya 距离 → 相似度
 *   身形：各几何维度的相对差异 → 相似度
 *
 * 另提供 averageFeatures()：对同一目标的多帧特征做逐元素平均（多帧累积），
 * 以提升比对鲁棒性。
 */
class ReIDExtractor
{
public:
    ReIDExtractor();

    /// 单帧特征提取（颜色+纹理+身形，1×TOTAL_DIM）
    cv::Mat extract(const cv::Mat &frame, const cv::Rect &personBox);

    /// 多帧累积：对多帧特征按分量求平均（尺寸/通道不一致的自动跳过），返回 1×N
    static cv::Mat averageFeatures(const QVector<cv::Mat> &features);

    /**
     * @brief 多帧时序融合（指数滑动平均 EMA）
     *
     * 越靠后的帧权重越高：fused = α·cur + (1-α)·prev，兼顾稳定性与时效性，
     * 相比等权平均更贴近“最近状态”，且天然抑制偶发噪声帧。
     * @param features 按时间顺序排列的特征序列
     * @param alpha    新帧权重（0.2~0.5 较合适），<=0 时退化为等权平均
     */
    static cv::Mat temporalFuse(const QVector<cv::Mat> &features, float alpha = 0.4f);

    /// 加权融合相似度 [0,1]
    float compare(const cv::Mat &feat1, const cv::Mat &feat2);

    /// 分量级评分（供动态加权/全局匹配使用）
    ComponentScores compareComponents(const cv::Mat &feat1, const cv::Mat &feat2) const;

    /// 指定权重下的融合相似度（不修改成员权重）
    float compareWithWeights(const cv::Mat &feat1, const cv::Mat &feat2,
                             float wColor, float wTexture, float wShape) const;

    /// 完整特征维度（手工特征；供外部校验/迁移旧特征时使用）
    static int featureDim() { return TOTAL_DIM; }

    // ===== 深度 ReID 后端（OSNet x1.0，可选）=====
    /**
     * @brief 绑定深度 ReID 后端（非拥有）
     * @param deep 深度模型实例（需已 load 成功；可为 nullptr 关闭）
     * @param use  true 且模型就绪时，extract()/compare() 改走深度特征（512 维余弦）
     *
     * 说明：俯视场景下手工颜色/纹理区分力弱，现场微调的 OSNet 深度嵌入
     * （rank-1 0.98、best-second gap 0.55）显著更可靠，故默认启用。
     */
    void setDeepReid(DeepReID *deep, bool use);
    /// 当前是否实际使用深度后端（绑定且就绪且启用）
    bool deepEnabled() const;
    /// 当前有效特征维度（深度=512，手工=featureDim()）
    int dim() const;

    /// 对任意维度特征做 L2 归一化（深度特征做余弦比较前必须归一化）
    static cv::Mat l2normalize(const cv::Mat &v);

    /// 设置三类分量权重（内部自动归一化）
    void setComponentWeights(float color, float texture, float shape);
    float weightColor()   const { return m_wColor; }
    float weightTexture() const { return m_wTexture; }
    float weightShape()   const { return m_wShape; }

    /// 绑定背景模型：绑定后特征提取只统计人体前景像素、剔除背景（传 nullptr 关闭）
    void setBackgroundModel(BackgroundModel *model) { m_bgModel = model; }
    BackgroundModel *backgroundModel() const { return m_bgModel; }

private:
    // ---- 颜色：HSV 三维直方图（上/下半身各一份）----
    // 分箱过细会使直方图在小面积人体上塌陷为噪声，这里取较粗的分辨率：
    //   H: 15 箱（12°/箱）、S: 8 箱、V: 8 箱 → 每半身 960 箱
    static constexpr int H_BINS = 15;
    static constexpr int S_BINS = 8;
    static constexpr int V_BINS = 8;
    static constexpr int COLOR_HALF = H_BINS * S_BINS * V_BINS;   // 960
    static constexpr int COLOR_DIM  = COLOR_HALF * 2;             // 1920

    // ---- 纹理：灰度梯度方向直方图（上/下半身各一份）----
    static constexpr int TEX_BINS = 18;
    static constexpr int TEX_DIM  = TEX_BINS * 2;                 // 36

    // ---- 身形：6 维几何/体积代理 ----
    //   0 长宽比(w/h) 1 归一化高度(h/H) 2 归一化宽度(w/W)
    //   3 面积占比    4 前景填充率    5 肩部宽度比(最宽肩宽/h)
    static constexpr int SHAPE_DIM = 6;

    // 完整特征维度
    static constexpr int TOTAL_DIM = COLOR_DIM + TEX_DIM + SHAPE_DIM; // 1962

    cv::Mat computeColorHistogram(const cv::Mat &roi, const cv::Mat &mask);
    cv::Mat computeTextureHistogram(const cv::Mat &roi, const cv::Mat &mask);
    static cv::Mat computeShapeVec(const cv::Mat &frame, const cv::Rect &box,
                                   const cv::Mat &mask, int fgPixels);

    float compareColor(const cv::Mat &a, const cv::Mat &b) const;
    float compareTexture(const cv::Mat &a, const cv::Mat &b) const;
    float compareShape(const cv::Mat &a, const cv::Mat &b) const;

    // 颜色分量在鱼眼俯视 + 跨朝向场景下最不稳定，故降低其权重，
    // 提高对光照/朝向更鲁棒的纹理与身形权重。
    float m_wColor{0.30f};
    float m_wTexture{0.35f};
    float m_wShape{0.35f};

    BackgroundModel *m_bgModel{nullptr};   // 可选背景模型（前景分割）

    // ===== 深度 ReID 后端（非拥有）=====
    DeepReID *m_deep{nullptr};
    bool m_useDeep{false};

public:
    /**
     * @brief 头框 → 人体框 的估计扩展（俯视场景：头检测框偏小，深度 ReID 期望身体比例）
     *
     * expand<=1.0 表示不扩展。扩展以框中心为基准，按 1:2（宽:高）人形比例向下延展，
     * 使 OSNet 输入更接近其训练分布（visitor_V* 人体裁剪）。
     */
    void setBoxExpandForReid(float expand) { m_boxExpand = expand; }
    float boxExpandForReid() const { return m_boxExpand; }

private:
    float m_boxExpand{1.0f};
};

/**
 * @brief 多帧特征的时序融合缓存（按 key 维护 EMA 融合特征）
 *
 * 用途：对同一目标/乘客的连续多帧特征做递推平均，既避免保存整段序列带来的
 * 内存与计算开销，又能以 O(1) 得到稳定、可复用的“融合特征”。
 */
class TemporalFeatureCache
{
public:
    void setAlpha(float alpha) { m_alpha = std::max(0.05f, std::min(0.95f, alpha)); }

    /// 更新并返回该 key 的融合特征（首帧直接存入）
    cv::Mat update(int key, const cv::Mat &feat);

    /// 获取融合特征（不存在返回空）
    cv::Mat get(int key) const;

    /// 累计帧数（不存在返回 0）
    int count(int key) const;

    void remove(int key) { m_entries.remove(key); }
    void clear() { m_entries.clear(); }
    int  size() const { return m_entries.size(); }

private:
    struct Entry { cv::Mat fused; int n{0}; };
    QMap<int, Entry> m_entries;
    float m_alpha{0.4f};
};

#endif // REIDEXTRACTOR_H
