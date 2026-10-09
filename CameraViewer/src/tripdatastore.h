#ifndef TRIPDATASTORE_H
#define TRIPDATASTORE_H

/**
 * @file tripdatastore.h
 * @brief 乘车数据存储抽象层：记录「每一站的上下车人数」与「车厢内人数」
 *
 * 设计目标：
 *   1. 统一记录每一站的上车人数、下车人数、车厢内（在车）人数；
 *   2. 通过抽象接口屏蔽后端差异，默认使用内置 SQLite（零依赖、随程序落盘），
 *      并预留 PostgreSQL 后端以应对大规模车队（多车并发、集中存储）；
 *   3. 上层业务只依赖 TripDataStore，切换后端仅需：
 *          TripDataStore *s = TripDataStore::create("postgresql");
 *          s->open({ {"host","..."}, {"dbname","..."}, ... });
 *
 * 表结构（SQLite / PostgreSQL 通用）：
 *   trips           (trip_id PK, vehicle_id, started_at, ended_at)
 *   stop_stats      (trip_id, station_id, station_name, boarding, alighting, onboard,
 *                    updated_at, PRIMARY KEY(trip_id, station_id))
 *   passenger_events(seq 自增, trip_id, station_id, station_name, passenger_id,
 *                    event_type, recorded_at)
 */

#include <QString>
#include <QVariantMap>
#include <QDateTime>
#include <QVector>
#include <QMap>

/// 每一站的上下车统计（含车厢内人数快照）
struct StopStat {
    QString tripId;
    QString stationId;
    QString stationName;
    int boarding{0};
    int alighting{0};
    int onboard{0};
    QDateTime updatedAt;
};

class TripDataStore
{
public:
    virtual ~TripDataStore();

    /// 后端标识："sqlite" / "postgresql"
    virtual QString backendName() const = 0;
    virtual bool isOpen() const = 0;

    /// 打开连接并初始化表结构（options 含义随后端而异）
    virtual bool open(const QVariantMap &options, QString *error = nullptr) = 0;
    virtual void close() = 0;

    /// 开始 / 结束一趟行程
    virtual bool beginTrip(const QString &tripId, const QString &vehicleId,
                           const QDateTime &startTime, QString *error = nullptr) = 0;
    virtual bool endTrip(const QString &tripId, const QDateTime &endTime,
                         QString *error = nullptr) = 0;

    /// 记录一次上车（自动累加到「当前站」并更新车厢内人数）
    virtual bool recordBoarding(const QString &tripId, const QString &stationId,
                                const QString &stationName, const QString &passengerId,
                                const QDateTime &time, QString *error = nullptr) = 0;
    /// 记录一次下车
    virtual bool recordAlighting(const QString &tripId, const QString &stationId,
                                 const QString &stationName, const QString &passengerId,
                                 const QDateTime &time, QString *error = nullptr) = 0;

    /// 查询某行程所有站点统计（按更新时间升序）
    virtual QVector<StopStat> stopStats(const QString &tripId) const = 0;
    /// 当前车厢内人数
    virtual int onboardCount(const QString &tripId) const = 0;
    /// 当前站的累计（内存镜像）
    virtual StopStat currentStop(const QString &tripId) const = 0;

    /**
     * @brief 工厂：按后端创建实例
     * @param backend "sqlite"（默认）或 "postgresql"
     * @return 新建实例（调用方负责 delete）；未知后端回退 SQLite
     */
    static TripDataStore *create(const QString &backend = QStringLiteral("sqlite"));

protected:
    /// 内存累计：tripId + '\x1f' + stationId -> StopStat
    QMap<QString, StopStat> m_liveStops;
    /// 内存累计：tripId -> 车厢内人数
    QMap<QString, int> m_liveOnboard;

    static QString stopKey(const QString &tripId, const QString &stationId);
};

/**
 * @brief 基于 Qt SQL 的通用实现（SQLite / PostgreSQL 仅驱动与建表语句不同）
 */
class SqlTripStore : public TripDataStore
{
public:
    explicit SqlTripStore(const QString &driver, const QString &connectionName);
    ~SqlTripStore() override;

    QString backendName() const override { return m_backend; }
    bool isOpen() const override { return m_open; }

    bool open(const QVariantMap &options, QString *error = nullptr) override;
    void close() override;

    bool beginTrip(const QString &tripId, const QString &vehicleId,
                   const QDateTime &startTime, QString *error = nullptr) override;
    bool endTrip(const QString &tripId, const QDateTime &endTime,
                 QString *error = nullptr) override;

    bool recordBoarding(const QString &tripId, const QString &stationId,
                        const QString &stationName, const QString &passengerId,
                        const QDateTime &time, QString *error = nullptr) override;
    bool recordAlighting(const QString &tripId, const QString &stationId,
                         const QString &stationName, const QString &passengerId,
                         const QDateTime &time, QString *error = nullptr) override;

    QVector<StopStat> stopStats(const QString &tripId) const override;
    int onboardCount(const QString &tripId) const override;
    StopStat currentStop(const QString &tripId) const override;

private:
    bool createTables(QString *error);
    bool ensureTrip(const QString &tripId, QString *error);
    bool applyStopEvent(const QString &tripId, const QString &stationId,
                        const QString &stationName, const QString &passengerId,
                        const QString &eventType, const QDateTime &time,
                        QString *error);
    int  loadOnboardFromDb(const QString &tripId) const;

    QString m_driver;
    QString m_connName;
    QString m_backend;
    bool m_open{false};
};

#endif // TRIPDATASTORE_H
