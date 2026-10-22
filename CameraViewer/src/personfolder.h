#ifndef PERSONFOLDER_H
#define PERSONFOLDER_H

#include <QString>
#include <QMap>
#include <QSet>
#include <QVector>
#include <QDateTime>
#include <QJsonObject>
#include <opencv2/opencv.hpp>
#include <memory>

/**
 * @brief 按乘客分目录的图像归档管理器
 *
 * 目录结构（save_images 下）：
 *   save_images/
 *   ├── match.txt                 ← 乘客列表汇总：哪些上车、哪些下车、哪些已匹配
 *   ├── person_1/                 ← 第 1 个出现的乘客（会话级顺序编号）
 *   │   ├── 原始图片/             ← 出现期间持续保存的原始整帧（不裁剪）
 *   │   ├── 乘客图片/             ← 出现期间持续保存的人体裁剪原图
 *   │   └── match.txt             ← 该乘客的方向判定（上车/下车）与匹配结果
 *   ├── person_2/
 *   │   ├── 原始图片/
 *   │   ├── 乘客图片/
 *   │   └── match.txt
 *   └── ...
 *
 * 编号为「会话级外观序号」：乘客一出现即分配 person_N，直到其消失；下一位乘客累加。
 * 该编号与 ReID 身份无关——同一位乘客上车/下车可能是不同的 person_N，
 * 最终由 FolderMatchWorker 全图比对后写入 match.txt 把上/下车文件夹关联起来。
 *
 * 线程模型：仅在主线程（帧处理 + 上/下车事件回调 + 匹配结果回调）中调用。
 */
class PersonFolder
{
public:
    explicit PersonFolder(const QString &basePath);

    /// 开始新会话：清空映射与序号（已落盘目录保留）
    void reset();

    /// 连续保存节流：每 N 帧实际写入 1 组
    void setSaveEveryN(int n) { m_saveEveryN = qMax(1, n); }

    /// 确保某 track 已分配 person 编号并创建 person_N/{原始图片,乘客图片}；返回编号
    int ensurePerson(int trackId);
    /// 新 track 续接已有 person 编号（时间相邻且 ReID 相似时“不换号”）：
    /// 直接把 track 挂到 personId，沿用其目录与图片序号，不新增 person_N。
    int reusePerson(int trackId, int personId);
    /// 查询 track 对应的 person 编号；不存在返回 -1
    int personIdForTrack(int trackId) const;
    bool hasPerson(int trackId) const { return m_trackToPerson.contains(trackId); }

    /**
     * @brief 保存一帧：原始整帧 → 原始图片/，人体裁剪 → 乘客图片/（带节流）
     * @return 是否实际写入
     */
    bool saveFrames(int personId, const cv::Mat &fullFrame, const cv::Mat &crop,
                    const QDateTime &ts);

    /// 写入/更新该乘客的方向判定（上车/下车 + 时间），并刷新汇总 match.txt
    void setDirection(int personId, bool boarding, const QDateTime &time);

    /// 关闭该乘客的原始图片视频写入（穿越完成/乘客消失时调用）
    void closeVideo(int personId);

    /// 该乘客是否已完成匹配（去重用）
    bool isMatched(int personId) const { return m_matched.contains(personId); }

    /// 命中后写入双方 match.txt 的「匹配对象 + 相似度」，并刷新汇总 match.txt
    void markMatched(int boardPersonId, int alightPersonId,
                     float similarity, const QDateTime &time);

    QString personDir(int personId) const;
    QString personName(int personId) const;   // 例如 person_1_20261013_180200
    /// 导出会话状态（重启后恢复用）
    QJsonObject stateToJson() const;
    /// 恢复会话状态
    void stateFromJson(const QJsonObject &obj);
    QString rawImagesDir(int personId) const;
    QString cropImagesDir(int personId) const;
    QString markerPath(int personId) const;
    QString summaryPath() const;

    int nextPersonId() const { return m_nextPersonId; }
    QString basePath() const { return m_basePath; }

private:
    QString dirFor(int personId, const QString &sub) const;
    void writeMarker(int personId);
    void writeSummary();
    void ensureVideoOpen(int personId, const cv::Mat &frame);
    void closeAllVideos();

    QString m_basePath;
    int m_nextPersonId{1};
    int m_saveEveryN{3};

    QMap<int, int> m_trackToPerson;   // trackId -> person 编号
    QVector<int> m_personOrder;       // person 编号按出现顺序
    QMap<int, QString> m_personDate;  // person 编号 -> 出现日期时间 yyyyMMdd_HHmmss
    QMap<int, int> m_personTick;      // person 编号 -> 节流计数
    QMap<int, int> m_personSeq;       // person 编号 -> 图片序号

    QMap<int, bool> m_boarding;       // person 编号 -> true=上车, false=下车
    QMap<int, QDateTime> m_eventTime; // person 编号 -> 事件时间
    QMap<int, int> m_matchedWith;     // person 编号 -> 匹配到的对方 person 编号
    QMap<int, float> m_matchedSim;    // person 编号 -> 匹配相似度
    QSet<int> m_matched;              // 已完成匹配的 person 编号

    // 每个乘客一段「原始图片对应视频」，写入 person_N/原始图片/视频.mp4（回退 avi）
    QMap<int, std::shared_ptr<cv::VideoWriter>> m_writers;
    // 每个乘客的「原始图片视频」分段序号：消失后再次续接同一编号时，
    // 视频换用 视频_2.avi / 视频_3.avi …，避免覆盖此前已关闭的段落。
    QMap<int, int> m_videoSession;
};

#endif // PERSONFOLDER_H
