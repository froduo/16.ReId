#ifndef OFFLINEDETECTFORM_H
#define OFFLINEDETECTFORM_H

#include <QDialog>
#include <QCheckBox>
#include <QLabel>
#include <QPushButton>
#include <QTableWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFileDialog>
#include <QMap>
#include <QPair>
#include <QStringList>
#include <opencv2/opencv.hpp>

class QTimer;
#include "imageview.h"
#include "persondetector.h"
#include "persontracker.h"
#include "reidextractor.h"
#include "reiddeep.h"
#include "regionmanager.h"
#include "zonemonitor.h"

/**
 * @brief 离线图像检测窗口
 *
 * 除检测/跟踪外，复用 ZoneMonitor 的通道状态机对**多张连续图片**做方向判定：
 *   - 人从上往下穿过检测区域：进入侧=A(顶)、离开侧=B(底) → **下车**
 *   - 人从下往上穿过检测区域：进入侧=B(底)、离开侧=A(顶) → **上车**
 * 连续检测时按图片顺序逐帧喂给 ZoneMonitor（tracker 状态在多次检测间保持）。
 */
class OfflineDetectForm : public QDialog
{
    Q_OBJECT
public:
    explicit OfflineDetectForm(PersonDetector *detector,
                               RegionManager *regionManager,
                               bool useDeepReid = true,
                               const QString &deepModelPath = QString(),
                               float deepMatchThreshold = 0.50f,
                               QWidget *parent = nullptr);
    ~OfflineDetectForm();

private slots:
    void onLoadImage();     // 加载单张（或追加到序列）
    void onRunDetect();     // 检测当前图片
    void onBatchDetect();   // 连续检测全部已加载图片（方向判定）
    void onVideoDetect();   // 加载一段视频并判定「上车 / 下车」（边播放边检测）
    void onVideoTick();     // 定时器：读取并显示一帧，同时做检测/方向判定
    void onVideoStop();     // 手动停止视频判定并给出当前结论

    // ===== 上/下车图像相似度比对（判定是否同一乘客） =====
    void onLoadBoardingImages();  // 加载上车图像（1 张或多张）
    void onLoadAlightingImages(); // 加载下车图像（1 张或多张）
    void onCompareGroups();       // 比较两组图像相似度并判定

    // 上车/下车 预览窗体的翻页
    void onBoardPrev();
    void onBoardNext();
    void onAlightPrev();
    void onAlightNext();

    void onDeepReidToggled(bool on);  // 切换手工/深度特征（切换后需重新加载图像）

private:
    void setupUI();
    void processFrameInternal(const cv::Mat &frame); // 检测+跟踪+区域方向+显示
    /// 结束视频判定：停止定时器、释放视频、统计并给出结论
    void finishVideoDetect(const QString &reason);
    void displayResult(const cv::Mat &frame,
                       const QVector<cv::Rect> &detections,
                       const QVector<TrackedPerson> &trackedPersons);
    void drawRegionOverlay(cv::Mat &frame, int imgW, int imgH);
    void refreshZoneMonitor(); // 依据当前区域刷新 ZoneMonitor 通道定义
    /// 从单张图片提取“乘客特征”（优先取检测到的人体框，取不到则整图）
    cv::Mat imageFeature(const cv::Mat &img);
    /// 在指定 ImageView 中显示图片文件（自动 BGR->RGB）
    static void showImageFile(ImageView *view, const QString &path);
    /// 刷新上车/下车预览（按当前索引显示，并更新文件名标签）
    void showBoardPreview();
    void showAlightPreview();
    /// 两组特征序列相似度：对 A 组每张取与 B 组的最优相似度再平均（含180°旋转不变）
    float groupSimilarity(const QVector<cv::Mat> &a, const QVector<cv::Mat> &b) const;
    /// 判定阈值（深度模式用深度阈值，否则用手工 ReID 阈值）
    float matchThreshold() const { return m_useDeepReid ? m_deepThreshold : m_matchThreshold; }
    /// 两组特征均值（L2 归一化，供深度模式质心比对）
    static cv::Mat meanFeature(const QVector<cv::Mat> &feats);

    ImageView *m_imageView{nullptr};
    QPushButton *m_btnLoad{nullptr};
    QPushButton *m_btnDetect{nullptr};
    QPushButton *m_btnBatch{nullptr};
    QPushButton *m_btnVideo{nullptr};        // 加载视频判定上/下车
    QPushButton *m_btnVideoStop{nullptr};    // 停止视频判定
    QLabel *m_labelInfo{nullptr};
    QLabel *m_labelVideoResult{nullptr};     // 视频判定结论
    QTableWidget *m_tableResults{nullptr};

    // 上/下车图像比对
    QPushButton *m_btnLoadBoard{nullptr};
    QPushButton *m_btnLoadAlight{nullptr};
    QPushButton *m_btnCompare{nullptr};
    QLabel *m_labelCompare{nullptr};
    QLabel *m_labelBoardCount{nullptr};
    QLabel *m_labelAlightCount{nullptr};

    // 上车/下车 图像预览窗体
    ImageView *m_boardView{nullptr};
    ImageView *m_alightView{nullptr};
    QLabel *m_labelBoardImage{nullptr};
    QLabel *m_labelAlightImage{nullptr};
    QPushButton *m_btnBoardPrev{nullptr};
    QPushButton *m_btnBoardNext{nullptr};
    QPushButton *m_btnAlightPrev{nullptr};
    QPushButton *m_btnAlightNext{nullptr};
    int m_boardPreviewIndex{0};       // 当前显示的上车图像索引（默认 0=第一张）
    int m_alightPreviewIndex{0};      // 当前显示的下车图像索引（默认 0=第一张）
    QStringList m_boardPaths;         // 上车图像路径
    QStringList m_alightPaths;        // 下车图像路径
    QVector<cv::Mat> m_boardFeats;    // 上车图像特征
    QVector<cv::Mat> m_alightFeats;   // 下车图像特征
    float m_matchThreshold{0.65f};    // 手工特征判定阈值

    // 深度 ReID（OSNet/ONNX）
    QCheckBox *m_checkDeepReid{nullptr};
    DeepReID m_deepReid;
    bool m_useDeepReid{false};
    float m_deepThreshold{0.50f};     // 深度特征余弦判定阈值

    PersonDetector *m_detector{nullptr};
    PersonTracker *m_tracker{nullptr};
    ReIDExtractor *m_reidExtractor{nullptr};
    RegionManager *m_regionManager{nullptr};
    ZoneMonitor *m_zoneMonitor{nullptr};

    cv::Mat m_currentImage;
    QString m_currentImagePath;

    // 多张连续图片
    QStringList m_imagePaths;
    int m_currentIndex{-1};

    // 方向判定结果
    QMap<int, QString> m_trackDirection; // trackId -> "上车(从下往上)" / "下车(从上往下)"
    int m_boardCount{0};
    int m_alightCount{0};

    // ===== 视频判定（边播边检测）=====
    QTimer *m_videoTimer{nullptr};
    cv::VideoCapture m_videoCap;
    QString m_videoPath;
    int m_videoStep{1};          // 抽帧步长（限制每秒刷新/检测帧数）
    double m_videoFps{25.0};
    int m_videoFrameIdx{0};      // 已读取的原始帧序号
    int m_videoProcessed{0};     // 已处理（检测）的帧数
    bool m_videoActive{false};

    // 人物轨迹范围（归一化）：用于与检测区域对比，判断区域是否覆盖通道
    float m_trajMinCx{2.0f};
    float m_trajMaxCx{-1.0f};
    float m_trajMinCy{2.0f};
    float m_trajMaxCy{-1.0f};

    // 每条轨迹（trackId）的 y 范围，用于结论中确认“是否被拆成多个 ID”
    QMap<int, QPair<float, float>> m_trackYRange;
    // 每条轨迹的首/末 y（时间序），用于「方向性判定」：片段只包含部分穿越过程时，
    // 严格完整穿越判据会失败，此时按净位移方向给出方向性结论。
    QMap<int, QPair<float, float>> m_trackFirstLast;
};

#endif // OFFLINEDETECTFORM_H
