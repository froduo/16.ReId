#ifndef VISITORLOG_H
#define VISITORLOG_H

#include <QString>
#include <QMap>
#include <QVector>
#include <QDateTime>
#include <QJsonObject>
#include <QJsonArray>
#include <QJsonDocument>
#include <QFile>
#include <QDir>
#include <QCoreApplication>
#include <opencv2/opencv.hpp>

/**
 * @brief 访客记录持久化管理器
 *
 * 功能：
 * 1. 将访客记录（VisitorRecord）保存为 JSON 文件
 * 2. 启动时加载历史访客记录，恢复任务状态
 * 3. 记录进入/离开事件日志（CSV 格式，方便后续分析）
 * 4. 定期自动保存，防止意外中断丢失数据
 */
class VisitorLog
{
public:
    VisitorLog();

    /**
     * @brief 设置保存目录（JSON 和事件日志都保存在此目录下）
     */
    void setSaveDir(const QString &dir);

    /**
     * @brief 获取当前保存目录
     */
    QString saveDir() const { return m_saveDir; }

    // ===== 人员进出记录结构（与 MainWindow 的 VisitorRecord 对应）=====
    // 记录格式：(乘客ID, 上车时间, 下车时间)
    struct VisitorEntry {
        int visitorId;          // 乘客ID
        QString boardTime;      // 上车时间（ISO 8601，乘客从A端顶端→B端底端）
        QString alightTime;     // 下车时间（ISO 8601，空表示尚未下车）
        bool isInside;          // 是否仍在车上（已上车未下车）
        int regionId;
        int currentTrackId;     // -1 表示无
        // ReID 特征以 Base64 编码存储
        QString reidFeatureBase64;
        // 上车/下车抓拍图片路径
        QString enterImagePath; // = 上车图片路径
        QString exitImagePath;  // = 下车图片路径
        // 方向："boarding"(上车:从下往上), "alighting"(下车:从上往下), ""(未确定)
        QString direction;
    };

    // ===== 事件日志条目 =====
    struct EventEntry {
        QDateTime timestamp;
        QString eventType;      // "ENTER", "EXIT", "REENTRY", "BOARDING", "ALIGHTING", "SESSION_START", "SESSION_END"
        int visitorId;
        int trackId;
        int regionId;
        float similarity;       // ReID 匹配相似度（仅 REENTRY 时有效）
        QString details;
    };

    /**
     * @brief 保存所有访客记录到 JSON 文件
     * @param visitors 访客 ID -> VisitorEntry 映射
     */
    void saveVisitors(const QMap<int, VisitorEntry> &visitors);

    /**
     * @brief 从 JSON 文件加载访客记录
     * @return 访客 ID -> VisitorEntry 映射
     */
    QMap<int, VisitorEntry> loadVisitors();

    /**
     * @brief 记录一个事件到事件日志
     */
    void logEvent(const EventEntry &entry);

    /**
     * @brief 记录会话开始
     */
    void logSessionStart();

    /**
     * @brief 记录会话结束
     */
    void logSessionEnd();

    /**
     * @brief 获取下一个可用的访客 ID（基于历史记录）
     */
    int getNextVisitorId() const;

    /**
     * @brief 保存 ReID 特征为 Base64 字符串
     */
    static QString matToBase64(const cv::Mat &mat);

    /**
     * @brief 从 Base64 字符串恢复 ReID 特征
     */
    static cv::Mat base64ToMat(const QString &base64);

private:
    QString m_saveDir;
    QString m_visitorJsonPath;
    QString m_eventLogPath;

    void ensureDirExists(const QString &dir);
    QJsonObject visitorToJson(const VisitorEntry &entry) const;
    VisitorEntry jsonToVisitor(const QJsonObject &obj) const;
};

#endif // VISITORLOG_H
