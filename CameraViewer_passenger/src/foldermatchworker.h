#ifndef FOLDERMATCHWORKER_H
#define FOLDERMATCHWORKER_H

#include <QObject>
#include <QVector>
#include <QString>
#include <opencv2/opencv.hpp>

class DeepReID;

/**
 * @brief 上车目录候选（供下车目录比对）
 */
struct BoardFolderCandidate {
    int personId{-1};      // 上车外观编号 person_N
    QString dir;           // 上车目录绝对路径
    qint64 timeMs{0};      // 上车时间（毫秒，用于时间过滤）
};

/**
 * @brief 后台「整目录全图」乘客身份比对工作器
 *
 * 触发时机：一次下车事件归档完成后，把该下车站目录与此前所有「未匹配」的
 * 上车目录做全图深度 ReID 比对（准确率优先，实时性不要求）。
 * 每张图提取 512 维 OSNet 特征，目录级相似度采用 best-avg：
 *   对下车的每张图，取与上车目录所有图的最大余弦相似度，再平均。
 *
 * 线程安全：submit() 通过 Qt::QueuedConnection 投递到本对象所属线程串行执行；
 * DeepReID 内部自带互斥锁，可跨线程共享。
 */
class FolderMatchWorker : public QObject
{
    Q_OBJECT
public:
    explicit FolderMatchWorker(QObject *parent = nullptr);

    /// 绑定深度 ReID 后端（实际赋值经队列投递到本对象线程）
    void setDeepReid(DeepReID *deep, bool use);

    /// 每个目录最多读取的图像张数（准确率优先，默认 200）
    void setMaxImagesPerFolder(int n);

    /**
     * @brief 提交一次「下车站目录 vs 多个上车目录」的比对
     * @param jobId            任务号（原样回传）
     * @param alightPersonId   下车外观编号 person_N
     * @param alightDir        下车站目录（已归档）
     * @param candidates       候选上车目录（由调用方过滤：时间更早 + 未匹配）
     * @param threshold        判定阈值 [0,1]
     */
    void submit(quint64 jobId, int alightPersonId, const QString &alightDir,
                const QVector<BoardFolderCandidate> &candidates, float threshold);

signals:
    /// 单次比对完成：bestPersonId=-1 表示无候选/未匹配
    void finished(quint64 jobId, int bestPersonId, QString bestDir,
                  float similarity, bool matched);

private:
    void run(quint64 jobId, int alightPersonId, const QString &alightDir,
             const QVector<BoardFolderCandidate> &candidates, float threshold);

    static QStringList imageFiles(const QString &dir, int maxCount);
    static QVector<cv::Mat> embedAll(DeepReID *deep, const QString &dir, int maxCount);
    static float bestAvgSimilarity(DeepReID *deep,
                                   const QVector<cv::Mat> &probe,
                                   const QVector<cv::Mat> &templ);

    DeepReID *m_deep{nullptr};
    bool m_useDeep{false};
    int m_maxImages{200};
};

#endif // FOLDERMATCHWORKER_H
