#ifndef PERSONFOLDER_H
#define PERSONFOLDER_H

#include <QString>
#include <QMap>
#include <QSet>
#include <QVector>
#include <QDateTime>
#include <QJsonObject>
#include <QMutex>
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
    /// visitorId>0 时优先按 ReID 身份复用当前活跃周期内的编号（轨迹断裂恢复），
    /// 避免同一乘客一次穿越运动因 track 断裂被拆成多个 person_N。
    int ensurePerson(int trackId, int visitorId = -1);
    /// 每次穿越事件后调用：为 track 分配新的 person 编号（person_N +1），
    /// 使同一乘客的多次穿越各自独立归档、独立计数。
    int rotatePerson(int trackId, int visitorId = -1);
    /// 查询 track 对应的 person 编号；不存在返回 -1
    int personIdForTrack(int trackId) const;
    bool hasPerson(int trackId) const { return m_trackToPerson.contains(trackId); }
    /// 清空 track→person 映射：track ID 会随 PersonTracker 重建而复用，
    /// 若不清理，新 track 会误复用旧 person 编号（表现为 person_N 重新从 1 开始）。
    void clearTrackMappings() { m_trackToPerson.clear(); }

    /**
     * @brief 保存一帧：原始整帧 → 原始图片/，人体裁剪 → 乘客图片/（带节流）
     * @return 是否实际写入
     *
     * 节流与序号分配在主线程同步完成；实际 JPEG 编码 + 视频写盘投递到全局线程池
     * 异步执行（writeFramesAsync），避免主线程被磁盘 IO / 视频编码阻塞。
     */
    bool saveFrames(int personId, const cv::Mat &fullFrame, const cv::Mat &crop,
                    const QDateTime &ts);

    /// 后台线程实际写盘入口（由 QThreadPool 投递的 QRunnable 调用）
    void writeFramesAsync(int personId, const cv::Mat &fullFrame, const cv::Mat &crop,
                          const QString &rawDir, const QString &cropDir,
                          const QString &name);

    /// 写入/更新该乘客的方向判定（上车/下车 + 时间），并刷新汇总 match.txt
    void setDirection(int personId, bool boarding, const QDateTime &time);

    /// 查询该乘客是否已判定为「上车」（用于区分同一 track 的首次穿越与再次穿越）
    bool isBoarding(int personId) const { return m_boarding.value(personId, false); }

    /// 将乘客方向改写为「上车折返/下车折返」（第三/四状态），不计入上下车统计；
    /// 供 MainWindow 收到 ZoneMonitor 的无效事件时回滚调用。
    void setInvalid(int personId, bool wasBoarding, const QDateTime &time);

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

    /// 设置下一个 person 序号（视频模式切换时继承旧序号，避免与保留的人员记录编号冲突）
    void setNextPersonId(int id) { if (id >= 1) m_nextPersonId = id; }
    QString basePath() const { return m_basePath; }

private:
    QString dirFor(int personId, const QString &sub) const;
    void writeMarker(int personId);
    void writeSummary();
    /// 追加一条归档事件到「归档事件日志.txt」（记录 person/track/visitor 分配原因，便于排查分裂）
    void logEvent(int personId, int trackId, int visitorId, const QString &reason);
    /// 打开该乘客的原始图片视频写入器（调用者须已持有 m_ioMutex）
    void openVideoLocked(int personId, const cv::Mat &frame);
    void closeAllVideos();

    QString m_basePath;
    int m_nextPersonId{1};
    int m_saveEveryN{3};

    QMap<int, int> m_trackToPerson;   // trackId -> person 编号
    QMap<int, int> m_visitorToPerson; // visitorId -> person 编号（当前活跃出现周期内复用）
    QMap<int, int> m_personToVisitor; // personId -> visitorId（反向，穿越结算时解绑）
    QVector<int> m_personOrder;       // person 编号按出现顺序
    QMap<int, QString> m_personDate;  // person 编号 -> 出现日期时间 yyyyMMdd_HHmmss
    QMap<int, int> m_personTick;      // person 编号 -> 节流计数
    QMap<int, int> m_personSeq;       // person 编号 -> 图片序号

    QMap<int, bool> m_boarding;       // person 编号 -> true=上车, false=下车
    QMap<int, QString> m_invalid;     // person 编号 -> "上车折返"/"下车折返"（第三/四状态）
    QMap<int, QDateTime> m_eventTime; // person 编号 -> 事件时间
    QMap<int, int> m_matchedWith;     // person 编号 -> 匹配到的对方 person 编号
    QMap<int, float> m_matchedSim;    // person 编号 -> 匹配相似度
    QSet<int> m_matched;              // 已完成匹配的 person 编号

    // 每个乘客一段「原始图片对应视频」，写入 person_N/原始图片/视频.mp4（回退 avi）
    QMap<int, std::shared_ptr<cv::VideoWriter>> m_writers;

    // 已关闭视频写入的 person 编号：防止 closeVideo 后残留的异步任务重新打开 writer
    QSet<int> m_closedVideos;

    /// 保护 m_writers 及 m_personSeq/m_personTick（主线程与保存线程并发访问）
    mutable QMutex m_ioMutex;
};

#endif // PERSONFOLDER_H
